#!/usr/bin/env python3
"""
Dual Arm Oculus VR Server - For Jaka Robots (Deoxys-style Control)
- Controls Left & Right Jaka robots via Oculus Quest controllers.
- synchronized 14-DOF Action/Observation recording.
- Global camera handling.
"""

import time
import threading
import numpy as np
import signal
import sys
import argparse
import copy
import math
import queue
from dataclasses import dataclass
from typing import Dict, Optional, List
from scipy.spatial.transform import Rotation as R

# --- Import Custom Modules ---
# 确保这些模块在你的 PYTHONPATH 或当前目录下
from oculus_reader.reader import OculusReader
from Modules.jaka_control import JakaRobot
from Modules.mcap_data_recorder import MCAPDataRecorder
# from Modules.camera_manager import CameraManager # Lazy import in main class

# --- Constants & Configs ---
CONTROL_FREQ = 50   # Hz
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 1.0
SAFE_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 示例

# --- Math Helper Functions ---

def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X

def euler_to_quat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_quat()

def quat_to_euler(quat, degrees=False):
    return R.from_quat(quat).as_euler("xyz", degrees=degrees)

def compute_angle_increment(current_angles, reference_angles):
    """计算角度增量，处理周期性"""
    current = np.asarray(current_angles, dtype=np.float64)
    reference = np.asarray(reference_angles, dtype=np.float64)
    delta = current - reference
    # Normalize to [-pi, pi]
    delta = (delta + math.pi) % (2 * math.pi) - math.pi
    return delta

# --- Data Structures ---

@dataclass
class RobotState:
    """单个机器人的状态快照"""
    timestamp: float
    pos: np.ndarray          # [x, y, z] (m)
    quat: np.ndarray         # [x, y, z, w]
    euler: np.ndarray        # [r, p, y] (rad)
    gripper: float           # 0.0 (Open) - 1.0 (Closed)
    joint_positions: Optional[np.ndarray] # [j1...j6]

    def copy(self):
        return RobotState(
            timestamp=self.timestamp,
            pos=self.pos.copy() if self.pos is not None else None,
            quat=self.quat.copy() if self.quat is not None else None,
            euler=self.euler.copy() if self.euler is not None else None,
            gripper=self.gripper,
            joint_positions=self.joint_positions.copy() if self.joint_positions is not None else None
        )

@dataclass
class TimestepData:
    """MCAP 录制的一帧完整数据"""
    timestamp: float
    vr_poses: Dict
    vr_buttons: Dict
    left_state: RobotState
    right_state: RobotState
    action: np.ndarray                      # 14-dim: [Left(7), Right(7)]
    info: Dict
    images: Optional[Dict[str, np.ndarray]] # 全局图像字典

# --- Single Arm Controller ---

class SingleArmController:
    """
    独立控制单条机械臂的逻辑单元。
    负责：连接机器人、状态更新、VR->Robot 映射计算。
    """
    def __init__(self, side: str, ip_address: str, coord_transform=None, debug=False, performance=False):
        self.side = side  # "left" or "right"
        self.controller_key = "l" if side == "left" else "r"
        self.ip = ip_address
        self.debug = debug
        self.performance = performance
        
        # --- Control Parameters ---
        self.pos_gain = 18.0 if performance else 5.0
        self.rot_gain = 1.5 if performance else 2.0
        self.gripper_gain = 3.0
        
        self.max_lin_vel = 1.0
        self.max_rot_vel = 1.0
        
        # Velocity to Delta conversion limits
        self.max_lin_delta = 0.005 if performance else 0.004
        self.max_rot_delta = 0.008 if performance else 0.007
        
        # Coordinate Transformation
        # Default: VR[z, x, y] -> Robot[x, y, z] (Standard DROID mapping)
        reorder = coord_transform if coord_transform else [2, 1, -3, 4]
        self.global_to_env_mat = vec_to_reorder_mat(reorder)
        self.vr_to_global_mat = np.eye(4) # Calibration matrix
        
        # State Variables
        self.robot = None
        self.current_state: Optional[RobotState] = None
        
        self.origin_robot = None
        self.origin_vr = None
        self.reset_origin = True
        
        # Filtering & Logic
        self._last_vr_pos = None
        self.prev_grip = False
        self.prev_joy = False
        self.calibrating_fwd = False
        self.calib_start_pose = None
        self.vr_neutral_pose = None
        
        self._robot_gripper_state = 0 # 0=Open, 1=Closed
        self._last_trig_cmd = GRIPPER_OPEN

        # Connect
        self._init_robot()

    def _init_robot(self):
        if self.debug:
            print(f"[{self.side}] Debug mode: Robot Simulated.")
            return
            
        print(f"[{self.side}] Connecting to {self.ip}...")
        try:
            self.robot = JakaRobot(self.ip)
            if self.robot.connect(0.02)[0]:
                self.robot.power_on()
                self.robot.enable()
                self.robot.init_gripper(0x01)
                self.robot.set_gripper_params(position=1000, force=30, speed=50, block=1)
                self.robot.robot.servo_move_enable(True)
                print(f"[{self.side}] ? Ready.")
            else:
                print(f"[{self.side}] ? Connection Failed.")
        except Exception as e:
            print(f"[{self.side}] ? Exception: {e}")

    def update_state(self):
        """读取机器人当前状态 (Observation)"""
        now = time.time()
        if self.debug:
            # 模拟状态
            self.current_state = RobotState(now, np.zeros(3), np.array([0,0,0,1]), np.zeros(3), 0.0, np.zeros(6))
            return self.current_state

        try:
            # Jaka return mm, rad/deg. Converting to meters and rad
            cart = np.array(self.robot.current_cartesian)
            joints = np.array(self.robot.current_joints)
            
            pos = 0.001 * cart[:3]
            euler = cart[3:] # Assuming Jaka returns Rad here. If Deg, convert!
            quat = euler_to_quat(euler)
            
            self.current_state = RobotState(
                timestamp=now,
                pos=pos, quat=quat, euler=euler,
                gripper=self._robot_gripper_state,
                joint_positions=joints
            )
            return self.current_state
        except Exception as e:
            # print(f"[{self.side}] Read Error: {e}")
            return None

    def step_control(self, vr_poses: Dict, vr_buttons: Dict):
        """
        计算单步动作
        Returns: action (7 dim), info (dict)
        """
        action = np.zeros(7)
        info = {}
        
        if self.controller_key not in vr_poses:
            return action, info
            
        # 1. Inputs
        pose_raw = np.asarray(vr_poses[self.controller_key])
        
        key_prefix = self.controller_key.upper()
        btn_grip = vr_buttons.get(key_prefix + "G", False)
        btn_joy = vr_buttons.get(key_prefix + "J", False)
        trig_val = vr_buttons.get("rightTrig" if self.side == "right" else "leftTrig", [0.0])[0]
        
        # Edge Detection
        grip_pressed = btn_grip and not self.prev_grip
        joy_pressed = btn_joy and not self.prev_joy
        joy_released = not btn_joy and self.prev_joy
        
        # 2. Calibration (Forward Direction)
        self._handle_calibration(pose_raw, joy_pressed, joy_released)
        
        # 3. Gripper Logic (Toggle)
        is_closing = trig_val > 0.1
        gripper_cmd = GRIPPER_CLOSE if is_closing else GRIPPER_OPEN
        if gripper_cmd == GRIPPER_CLOSE and self._last_trig_cmd == GRIPPER_OPEN:
            # Toggle
            self._robot_gripper_state = 1 - self._robot_gripper_state
            if not self.debug:
                if self._robot_gripper_state == 0:
                    self.robot.set_gripper_params(position=1000, force=30, speed=50, block=0)
                else:
                    self.robot.grasp(timeout=5)
        self._last_trig_cmd = gripper_cmd
        
        # 4. Motion Control
        if grip_pressed:
            self.reset_origin = True
            
        if btn_grip and self.current_state:
            if not self.debug:
                self.robot.robot.servo_move_enable(True)
                
            # A. VR Transform
            vr_pos, vr_euler = self._transform_pose(pose_raw)
            
            # B. Origin Reset
            if self.reset_origin:
                self.origin_robot = {"pos": self.current_state.pos, "euler": self.current_state.euler}
                self.origin_vr = {"pos": vr_pos, "euler": vr_euler}
                self._last_vr_pos = vr_pos.copy()
                self.reset_origin = False
                
            # C. Calculate Action
            if self.origin_robot:
                # Position (with deadzone filtering)
                pos_delta = vr_pos - self._last_vr_pos
                pos_delta[np.abs(pos_delta) < 0.003] = 0.0 # Input deadzone
                curr_filtered_pos = self._last_vr_pos + pos_delta
                self._last_vr_pos = curr_filtered_pos
                
                # Relative offsets
                target_pos_offset = curr_filtered_pos - self.origin_vr["pos"]
                robot_pos_offset = self.current_state.pos - self.origin_robot["pos"]
                
                pos_err = target_pos_offset - robot_pos_offset
                pos_action = pos_err * self.pos_gain
                
                # Rotation (Euler diff)
                target_rot_offset = compute_angle_increment(vr_euler, self.origin_vr["euler"])
                robot_rot_offset = compute_angle_increment(self.current_state.euler, self.origin_robot["euler"])
                
                rot_err = target_rot_offset - robot_rot_offset
                rot_action = rot_err * self.rot_gain
                
                # Limits & Clipping
                lin_scale = np.linalg.norm(pos_action)
                if lin_scale > self.max_lin_vel: pos_action *= self.max_lin_vel / lin_scale
                
                rot_scale = np.linalg.norm(rot_action)
                if rot_scale > self.max_rot_vel: rot_action *= self.max_rot_vel / rot_scale
                
                # Gripper Action (P-control to target state)
                grip_act = (self._robot_gripper_state - self.current_state.gripper) * self.gripper_gain
                
                action = np.concatenate([pos_action, rot_action, [grip_act]])
                action = np.clip(action, -1.0, 1.0)
                
                # D. Send to Robot
                if not self.debug:
                    self._send_servo_cmd(pos_action, rot_action)

        elif not btn_grip and self.prev_grip:
            # Stop on release
            if not self.debug:
                self.robot.robot.servo_move_enable(False)
                self.robot.stop_motion()

        # Update History
        self.prev_grip = btn_grip
        self.prev_joy = btn_joy
        
        return action, info

    def _send_servo_cmd(self, pos_action, rot_action):
        """将标准化动作转换为 Jaka 伺服指令"""
        # Vel -> Delta
        d_pos = pos_action * self.max_lin_delta
        d_rot = rot_action * self.max_rot_delta
        
        # Meters -> mm
        cmd_pos = d_pos * 1000.0
        # Euler Rad -> Rad (Jaka usually takes rad for Servo)
        # Coordinate mapping for rotation might need adjustment per robot mounting
        # Example: Inverting Pitch/Roll if needed
        cmd_rot = np.array([-d_rot[0], -d_rot[1], d_rot[2]]) 
        
        cmd = np.concatenate([cmd_pos, cmd_rot])
        
        step = 2 if self.performance else 5
        try:
            self.robot.robot.servo_p_extend(cmd, move_mode=1, step_num=step)
        except: pass

    def _transform_pose(self, raw_mat):
        """VR Raw -> Robot Global Frame"""
        # 1. Axis Reordering
        t_mat = self.global_to_env_mat @ self.vr_to_global_mat @ raw_mat
        pos = t_mat[:3, 3]
        
        # 2. Rotation Mapping (Labelbox style / DROID style)
        if self.vr_neutral_pose is not None:
            # Relative rotation from neutral
            n_rot = R.from_matrix(self.vr_neutral_pose[:3, :3])
            c_rot = R.from_matrix(raw_mat[:3, :3])
            rel = n_rot.inv() * c_rot
            
            # Map VR axes to Robot axes (Ergonomic mapping)
            # VR Y (Roll) -> Robot Y (Pitch) inverted
            # VR X (Pitch) -> Robot X (Roll) inverted
            rv = rel.as_rotvec()
            ang = np.linalg.norm(rv)
            if ang > 1e-6:
                ax = rv / ang
                # Mapping: [-y, -x, -z]
                mapped_ax = np.array([-ax[1], -ax[0], -ax[2]])
                euler = R.from_rotvec(mapped_ax * ang).as_euler("xyz")
            else:
                euler = np.zeros(3)
        else:
            # Absolute fallback
            euler = R.from_matrix(t_mat[:3, :3]).as_euler("xyz")
            
        return pos, euler

    def _handle_calibration(self, pose, joy_press, joy_release):
        if joy_press:
            self.calibrating_fwd = True
            self.calib_start_pose = pose.copy()
            print(f"[{self.side}] Calibrating Forward...")
        elif joy_release and self.calibrating_fwd:
            self.calibrating_fwd = False
            start = self.calib_start_pose[:3, 3]
            end = pose[:3, 3]
            vec = end - start
            dist = np.linalg.norm(vec)
            
            if dist > 0.003:
                # Align movement vector to Robot +X
                fwd = vec / dist
                # ... (Standard rotation alignment math) ...
                # Simply setting neutral pose for now
                self.vr_neutral_pose = pose.copy()
                print(f"[{self.side}] ? Forward set (Dist: {dist*1000:.1f}mm)")
            else:
                self.vr_neutral_pose = pose.copy()
                print(f"[{self.side}] Reset orientation.")

# --- Dual Arm Server ---

class DualOculusServer:
    def __init__(self, args):
        self.running = True
        self.debug = args.debug
        self.enable_recording = not args.no_recording
        
        # 1. Hardware Initialization
        print("? Init Oculus Reader...")
        self.oculus = OculusReader(ip_address=args.ip, print_FPS=False)
        
        print("\n? Init Robots...")
        # Note: Coordinate transforms might need to be different for left/right
        # depending on physical mounting. Using default for now.
        self.left_arm = SingleArmController("left", args.left_ip, debug=args.debug, performance=args.performance)
        self.right_arm = SingleArmController("right", args.right_ip, debug=args.debug, performance=args.performance)
        
        # 2. Global Camera Manager
        self.camera_manager = None
        if self.enable_recording and args.enable_cameras:
            try:
                from Modules.camera_manager import CameraManager
                print(f"? Init Camera Manager ({args.camera_config})...")
                self.camera_manager = CameraManager(args.camera_config)
            except ImportError:
                print("?? Camera module not found.")

        # 3. Data Recorder
        self.recorder = None
        if self.enable_recording:
            # We pass save_images=True, but we'll feed images manually
            self.recorder = MCAPDataRecorder(save_images=True, save_depth=False)
            print("? MCAP Recorder ready.")
            
        # 4. Threading
        self.recording_active = False
        self.mcap_queue = queue.Queue(maxsize=50)
        self.lock = threading.Lock()
        self.latest_data = None # Global state cache
        self.threads = []
        
        # Button state
        self.prev_a = False

        # Signal handlers
        signal.signal(signal.SIGINT, self.shutdown)

    def start(self):
        if self.camera_manager:
            self.camera_manager.start()
            
        # Start Threads
        self.threads.append(threading.Thread(target=self._control_loop, daemon=True))
        self.threads.append(threading.Thread(target=self._recording_worker, daemon=True))
        self.threads.append(threading.Thread(target=self._writer_worker, daemon=True))
        
        for t in self.threads: t.start()
        
        print("\n? System Running!")
        print("   [Right A]: Start/Discard Recording")
        print("   [Right B]: Save Recording")
        
        while self.running:
            time.sleep(1)

    def shutdown(self, sig, frame):
        print("\n? Shutting down...")
        self.running = False
        if self.camera_manager: self.camera_manager.stop()
        if not self.debug:
            try:
                self.left_arm.robot.robot.servo_move_enable(False)
                self.right_arm.robot.robot.servo_move_enable(False)
            except: pass
        sys.exit(0)

    def _control_loop(self):
        """High-frequency control loop (Robot Actuation)"""
        hz = CONTROL_FREQ * 2 if self.left_arm.performance else CONTROL_FREQ
        dt = 1.0 / hz
        
        while self.running:
            t0 = time.time()
            
            # 1. Read VR
            poses, buttons = self.oculus.get_transformations_and_buttons()
            
            # 2. Update Robots
            l_state = self.left_arm.update_state()
            r_state = self.right_arm.update_state()
            
            # 3. Control Logic
            if l_state and r_state:
                # Calculate Actions
                l_act, l_info = self.left_arm.step_control(poses, buttons)
                r_act, r_info = self.right_arm.step_control(poses, buttons)
                
                # Handle Global Recording Buttons (Right Controller)
                self._handle_rec_buttons(buttons)
                
                # Update Cache for Recorder
                with self.lock:
                    self.latest_data = {
                        "ts": t0,
                        "poses": copy.deepcopy(poses),
                        "buttons": copy.deepcopy(buttons),
                        "l_state": l_state.copy(),
                        "r_state": r_state.copy(),
                        "l_act": l_act,
                        "r_act": r_act,
                        "info": {"l": l_info, "r": r_info}
                    }
            
            # Sleep
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def _handle_rec_buttons(self, btns):
        if not self.enable_recording: return
        
        a_btn = btns.get("A", False)
        b_btn = btns.get("B", False)
        
        # A: Toggle (Start / Discard)
        if a_btn and not self.prev_a:
            if self.recording_active:
                print("? Stopped (Discarded).")
                self.recorder.reset_recording()
                self.recording_active = False
            else:
                print("? Started Recording.")
                self.recorder.start_recording()
                self.recording_active = True
        self.prev_a = a_btn
        
        # B: Save
        if b_btn and self.recording_active:
            path = self.recorder.stop_recording(success=True)
            self.recording_active = False
            print(f"? Saved: {path}")

    def _recording_worker(self):
        """Data Capture Loop (Syncs Robot State + Global Images)"""
        rec_dt = 1.0 / CONTROL_FREQ
        
        while self.running:
            start = time.time()
            
            if self.recording_active and self.latest_data:
                # 1. Capture Global Images NOW (minimize sync offset)
                imgs = {}
                if self.camera_manager:
                    imgs = self.camera_manager.get_frames() # Returns dict {id: np.array}
                
                # 2. Get Robot Data
                with self.lock:
                    d = copy.deepcopy(self.latest_data)
                
                # 3. Combine Action (14 dim)
                full_action = np.concatenate([d["l_act"], d["r_act"]])
                
                # 4. Pack
                ts_data = TimestepData(
                    timestamp=d["ts"],
                    vr_poses=d["poses"],
                    vr_buttons=d["buttons"],
                    left_state=d["l_state"],
                    right_state=d["r_state"],
                    action=full_action,
                    info=d["info"],
                    images=imgs # Global images attached here
                )
                
                try:
                    self.mcap_queue.put_nowait(ts_data)
                except queue.Full:
                    pass
            
            elapsed = time.time() - start
            if elapsed < rec_dt:
                time.sleep(rec_dt - elapsed)

    def _writer_worker(self):
        """Disk IO Loop"""
        while self.running:
            try:
                item = self.mcap_queue.get(timeout=0.1)
                
                # Format for MCAP (DROID / OpenX style)
                
                # Combine Joints (Flattened)
                joints = []
                if item.left_state.joint_positions is not None:
                    joints.extend(item.left_state.joint_positions)
                if item.right_state.joint_positions is not None:
                    joints.extend(item.right_state.joint_positions)
                
                # Combine Cartesian
                cart = np.concatenate([
                    item.left_state.pos, item.left_state.euler,
                    item.right_state.pos, item.right_state.euler
                ])
                
                timestep = {
                    "observation": {
                        "timestamp": int(item.timestamp * 1e9),
                        "robot_state": {
                            "joint_positions": joints,
                            "cartesian_position": cart.tolist(),
                            "gripper_position": [item.left_state.gripper, item.right_state.gripper]
                        },
                        # Global images passed directly
                        "images": item.images if item.images else {}
                    },
                    "action": item.action.tolist()
                }
                
                self.recorder.write_timestep(timestep, item.timestamp)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Write Error: {e}")

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left-ip', required=True, help='Left Robot IP')
    parser.add_argument('--right-ip', required=True, help='Right Robot IP')
    parser.add_argument('--ip', default=None, help='Oculus IP')
    parser.add_argument('--debug', action='store_true', help='Simulate robots')
    parser.add_argument('--performance', action='store_true', help='High Hz mode')
    parser.add_argument('--no-recording', action='store_true')
    parser.add_argument('--enable-cameras', action='store_true')
    parser.add_argument('--camera-config', default='configs/cameras.yaml')
    args = parser.parse_args()
    
    server = DualOculusServer(args)
    server.start()

if __name__ == "__main__":
    main()