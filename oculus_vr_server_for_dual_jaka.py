#!/usr/bin/env python3
"""
Oculus VR Server - Implements VRPolicy-style teleoperation control
Based on droid/controllers/oculus_controller.py

VR-to-Robot Control Pipeline:
1. VR Data Capture: Raw poses from Oculus Reader (50Hz internal thread)
2. Coordinate Transform: Apply calibrated transformation [X,Y,Z] → [-Y,X,Z]
3. Velocity Calculation: Position/rotation offsets with gains (pos=5, rot=2)
4. Velocity Limiting: Clip to [-1, 1] range
5. Delta Conversion: Scale by max_delta (0.075m linear, 0.15rad angular)
6. Position Target: Add deltas to current position/orientation
7. Deoxys Command: Send position + quaternion targets (15Hz)

Key Differences from DROID:
- DROID uses Polymetis (euler angles) vs our Deoxys (quaternions)
- We skip IK solver (Deoxys handles internally)
- Direct quaternion calculation for accurate rotation control
- VR motions map directly to robot motions (roll inverted for ergonomics)

Features:
- Velocity-based control with DROID-exact parameters
- Intuitive forward direction calibration (hold joystick + move forward)
- Origin calibration on grip press/release
- Coordinate transformation pipeline matching DROID
- Success/failure buttons (A/B or X/Y)
- 50Hz VR polling with internal state thread
- Safety limiting and workspace bounds
- Deoxys-compatible quaternion handling
- FR3 robot simulation mode
- MCAP data recording with Labelbox Robotics format
- Asynchronous architecture for high-performance recording
"""

# import zmq
import time
import threading
import numpy as np
import signal
import sys
import argparse
import pickle
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional, Tuple
import os
import queue
from dataclasses import dataclass
from collections import deque
import copy

# Import the Oculus Reader
from oculus_reader.reader import OculusReader
import math
# Import robot control components
from constants import (
    HOST, CONTROL_PORT,
    GRIPPER_OPEN, GRIPPER_CLOSE,
    ROBOT_WORKSPACE_MIN, ROBOT_WORKSPACE_MAX,
    CONTROL_FREQ, SAFE_POSITION # Use the constant from the config
)
from Modules.jaka_control import *




# Import MCAP data recorder
from Modules.mcap_data_recorder import MCAPDataRecorder
from Modules.mcap_verifier import MCAPVerifier



@dataclass
class VRState:
    """Thread-safe VR controller state"""
    timestamp: float
    poses: Dict
    buttons: Dict
    movement_enabled: bool
    controller_on: bool
    grip_released: bool
    toggle_state: bool
    def copy(self):
        """Deep copy for thread safety"""
        return VRState(
            timestamp=self.timestamp,
            poses=copy.deepcopy(self.poses),
            buttons=copy.deepcopy(self.buttons),
            movement_enabled=self.movement_enabled,
            controller_on=self.controller_on,
            grip_released=self.grip_released,
            toggle_state=self.toggle_state
        )


@dataclass
class RobotState:
    """Thread-safe robot state"""
    timestamp: float
    pos: np.ndarray
    quat: np.ndarray
    euler: np.ndarray
    gripper: float
    joint_positions: Optional[np.ndarray]
    
    def copy(self):
        """Deep copy for thread safety"""
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
    """Data structure for MCAP recording"""
    timestamp: float
    vr_state: VRState
    robot_state: RobotState
    action: np.ndarray
    info: Dict


def vec_to_reorder_mat(vec):
    """Convert reordering vector to transformation matrix"""
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X


def rmat_to_quat(rot_mat):
    """Convert rotation matrix to quaternion (x,y,z,w)"""
    rotation = R.from_matrix(rot_mat)
    return rotation.as_quat()


def quat_to_rmat(quat):
    """Convert quaternion (x,y,z,w) to rotation matrix"""
    return R.from_quat(quat).as_matrix()


def quat_diff(target, source):
    """Calculate quaternion difference"""
    result = R.from_quat(target) * R.from_quat(source).inv()
    return result.as_quat()


def quat_to_euler(quat, degrees=False):
    """Convert quaternion to euler angles"""
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_quat(euler, degrees=False):
    """Convert euler angles to quaternion"""
    return R.from_euler("xyz", euler, degrees=degrees).as_quat()


def add_angles(delta, source, degrees=False):
    """Add two sets of euler angles"""
    delta_rot = R.from_euler("xyz", delta, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    new_rot = delta_rot * source_rot
    return new_rot.as_euler("xyz", degrees=degrees)

def compute_angle_increment(current_angles, reference_angles, is_radians=True):
    """Calculate minimal angle difference"""
    current = np.asarray(current_angles, dtype=np.float64)
    reference = np.asarray(reference_angles, dtype=np.float64)
    
    delta = current - reference
    
    if is_radians:
        two_pi = 2 * math.pi
        limit = math.pi
    else:
        two_pi = 360.0
        limit = 180.0
    
    delta = (delta + limit) % two_pi - limit
    delta = np.where(delta <= -limit, delta + two_pi, delta)
    return delta
class ArmController:
    """
    负责单臂的控制，状态更新，连接机器人，以及 VR->Robot 的映射计算。
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
        
        # Deadzones
        self.output_deadzone = 0.003
        self.output_euler_deadzone = math.radians(5)
        self.translation_deadzone = 0.005
        
        # Coordinate Transformation
        # Default: VR[z, x, y] -> Robot[x, y, z] (Standard DROID mapping logic from original file)
        # Original: rmat_reorder = [2, 1, -3, 4] -> Z, X, -Y
        reorder = coord_transform if coord_transform else [2, 1, -3, 4]
        self.global_to_env_mat = vec_to_reorder_mat(reorder)
        self.vr_to_global_mat = np.eye(4) # Calibration matrix
        self.spatial_coeff = 1.0
        
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
        
        self._robot_gripper_state = 0 # 0=Open, 1=Closed logic counter
        self._last_trig_val = 0.0

        # Hardware Step Num
        self.step_num = 2 if self.performance else 5

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
                self.robot.robot.set_tool_id(1)
                self.robot.set_tool_data(140)
                
                # Move to home first
                self.reset_to_home()
                
                self.robot.robot.servo_move_enable(True)
                print(f"[{self.side}] ? Ready.")
            else:
                print(f"[{self.side}] ? Connection Failed.")
        except Exception as e:
            print(f"[{self.side}] ? Exception: {e}")

    def reset_to_home(self):
        """Reset robot to home position"""
        if self.debug: return
        # Home position from original code
        target_pos = [355, -122, 356, math.radians(180.0), math.radians(0.0), math.radians(145.0)]
        self.robot.move_linear_extend(target_pos, _speed=200, _accel=100, TOL=0.1, _is_block=True, _move_mode=0)
        self.reset_origin = True # Force origin reset after home

    def update_state(self):
        """读取机器人当前状态 (Observation)"""
        now = time.time()
        
        # Fallback / Initial
        default_pos = np.array([0.4, 0.0, 0.3])
        default_euler = np.array([3.14, 0, 0])
        
        if self.debug:
            self.current_state = RobotState(
                now, default_pos, euler_to_quat(default_euler), default_euler, 
                float(self._robot_gripper_state), np.zeros(6)
            )
            return self.current_state

        try:
            # Jaka return mm, rad/deg. 
            cart = np.array(self.robot.current_cartesian)
            joints = np.array(self.robot.current_joints)
            
            # Convert mm to meters
            pos = 0.001 * cart[:3]
            euler = cart[3:] # Jaka uses Euler (RX, RY, RZ) in Radians usually
            quat = euler_to_quat(euler)
            print(f"    new_{self.side}_robot_state:{cart}")
 
            self.current_state = RobotState(
                timestamp=now,
                pos=pos, quat=quat, euler=euler,
                gripper=float(self._robot_gripper_state),
                joint_positions=joints
            )
            return self.current_state
        except Exception as e:
            # print(f"[{self.side}] Read Error: {e}")
            return None

    def process_control(self, vr_state: VRState):
        """
        计算单步动作并发送给机器人
        Returns: action (7 dim), info (dict)
        """
        action = np.zeros(7)
        info = {}
        
        if self.controller_key not in vr_state.poses:
            return action, info
            
        # 1. Inputs
        pose_raw = np.asarray(vr_state.poses[self.controller_key])
        buttons = vr_state.buttons
        
        key_prefix = self.controller_key.upper()
        btn_grip = buttons.get(key_prefix + "G", False)
        btn_joy = buttons.get(key_prefix + "J", False)
        trig_val = buttons.get("rightTrig" if self.side == "right" else "leftTrig", [0.0])[0]
        joy_axis_val = buttons.get("rightJS" if self.side == "right" else "leftJS", [0.0])[0]
        
        # Edge Detection
        joy_pressed = btn_joy and not self.prev_joy
        joy_released = not btn_joy and self.prev_joy
        
        # 2. Calibration (Forward Direction)
        self._handle_calibration(pose_raw, joy_pressed, joy_released)
        
        # 3. Gripper Logic (Toggle on press)
        # Logic: If trigger pressed (>0.1) and was not pressed before -> toggle
        is_pressed = trig_val > 0.1
        was_pressed = self._last_trig_val > 0.1
        
        if is_pressed and not was_pressed:
            # Rising edge of trigger
            self._robot_gripper_state = 1 - self._robot_gripper_state # Toggle 0/1
            print(f"Gripper Toggle: {self._robot_gripper_state}")
            if not self.debug:
                if self._robot_gripper_state == 0:
                    self.robot.set_gripper_params(position=1000, force=30, speed=50, block=0) # Open
                else:
                    self.robot.grasp(timeout=5) # Close
        self._last_trig_val = trig_val
        
        # 4. Independent RZ Control (Joystick Axis)
        if abs(joy_axis_val) > 0.5 and not self.debug:
             # This uses linear move, might block servo slightly, use with caution
             # Original code used move_linear_extend for this jog
             rz_step = math.radians(0.5) if joy_axis_val > 0 else math.radians(-0.5)
             rz_delta = np.asarray([0, 0, 0, 0, 0, rz_step])
             try:
                self.robot.move_linear_extend(rz_delta, _speed=200, _accel=100, TOL=0.1, _is_block=False, _move_mode=1)
             except: pass

        # 5. Motion Control (Servo)
        # Handle Grip Edge for Origin Reset
        if btn_grip and not self.prev_grip:
            self.reset_origin = True
            if not self.debug: self.robot.robot.servo_move_enable(True)
            
        if btn_grip and self.current_state:
            # A. VR Transform
            vr_pos, vr_euler, vr_quat = self._transform_pose(pose_raw)
            
            # B. Origin Reset
            if self.reset_origin:
                self.origin_robot = {"pos": self.current_state.pos, "euler": self.current_state.euler, "quat": self.current_state.quat}
                self.origin_vr = {"pos": vr_pos, "euler": vr_euler, "quat": vr_quat}
                self._last_vr_pos = vr_pos.copy()
                self.reset_origin = False
                print("? Origin calibrated")
                
            # C. Calculate Action
            if self.origin_robot:
                # --- Position Calculation ---
                # Apply translation deadzone filter
                pos_delta = vr_pos - self._last_vr_pos
                for i in range(3):
                    if abs(pos_delta[i]) < self.translation_deadzone:
                        pos_delta[i] = 0.0
                curr_filtered_pos = self._last_vr_pos + pos_delta
                self._last_vr_pos = curr_filtered_pos
                
                # Calculate offsets
                target_pos_offset = curr_filtered_pos - self.origin_vr["pos"]
                robot_pos_offset = self.current_state.pos - self.origin_robot["pos"]
                print(f"vr_offset: {target_pos_offset}")
                print(f"robot_offset: {robot_pos_offset}")
                pos_err = target_pos_offset - robot_pos_offset
                # Output Deadzone
                for i in range(3):
                    if abs(pos_err[i]) < self.output_deadzone: pos_err[i] = 0.0
                
                pos_action = pos_err * self.pos_gain
                
                # --- Rotation Calculation ---
                target_rot_offset = compute_angle_increment(vr_euler, self.origin_vr["euler"])
                robot_rot_offset = compute_angle_increment(self.current_state.euler, self.origin_robot["euler"])
                print(f"target_rot_offset: {target_rot_offset}")
                print(f"target_rot_offset: {target_rot_offset}")
                rot_err = target_rot_offset - robot_rot_offset
                # Output Euler Deadzone
                for i in range(3):
                    if abs(rot_err[i]) < self.output_euler_deadzone: rot_err[i] = 0.0
                    
                rot_action = rot_err * self.rot_gain
                
                # --- Limits & Clipping ---
                lin_scale = np.linalg.norm(pos_action)
                if lin_scale > self.max_lin_vel: pos_action *= self.max_lin_vel / lin_scale
                
                rot_scale = np.linalg.norm(rot_action)
                if rot_scale > self.max_rot_vel: rot_action *= self.max_rot_vel / rot_scale
                
                # --- Gripper Action (P-control to target state) ---
                # This is primarily for the action recording, since actual gripper is toggled above
                grip_act = (float(self._robot_gripper_state) * 1.5 - self.current_state.gripper) * self.gripper_gain
                
                action = np.concatenate([pos_action, rot_action, [self._robot_gripper_state]])
                action = np.clip(action, -1.0, 1.0)
                
                # Info for recording
                info = {
                    "target_pos": (pos_action/self.pos_gain + self.current_state.pos).tolist(),
                    "gripper_state": self._robot_gripper_state
                }
                
                # D. Send to Robot
                if not self.debug:
                    self._send_servo_cmd(pos_action, rot_action)

        elif not btn_grip and self.prev_grip:
            # Stop on release
            print("    Gripper released Stop!")
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
        
        # Euler Rad -> Rad
        # Apply coordinate mapping inversion for the robot command if necessary
        # Based on original code: [x, y, z, -rx, -ry, rz]
        cmd_rot = np.array([-d_rot[0], -d_rot[1], d_rot[2]]) 
        
        cmd = np.concatenate([cmd_pos, cmd_rot])
        
        try:
            self.robot.robot.servo_p_extend(cmd, move_mode=1, step_num=self.step_num)
        except Exception as e:
            print(f"Servo Error: {e}")

    def _transform_pose(self, raw_mat):
        """
        VR Raw -> Robot Global Frame 
        Returns: pos, euler, quat
        """
        # 1. Axis Reordering & Calibration
        t_mat = self.global_to_env_mat @ self.vr_to_global_mat @ raw_mat
        pos = self.spatial_coeff * t_mat[:3, 3]
        
        # 2. Rotation Mapping (DROID / Labelbox style)
        if self.vr_neutral_pose is not None:
            # Relative rotation from neutral
            n_rot = R.from_matrix(self.vr_neutral_pose[:3, :3])
            c_rot = R.from_matrix(raw_mat[:3, :3])
            rel = n_rot.inv() * c_rot
            
            rv = rel.as_rotvec()
            ang = np.linalg.norm(rv)
            if ang > 1e-6:
                ax = rv / ang
                # Mapping: [-y, -x, -z] (Swap X/Y and negate Y)
                mapped_ax = np.array([-ax[1], -ax[0], -ax[2]])
                
                mapped_rot = R.from_rotvec(mapped_ax * ang)
                quat = mapped_rot.as_quat()
                euler = mapped_rot.as_euler("xyz")
            else:
                quat = np.array([0, 0, 0, 1])
                euler = np.zeros(3)
        else:
            # Absolute fallback
            rot_mat_t = t_mat[:3, :3]
            quat = R.from_matrix(rot_mat_t).as_quat()
            euler = R.from_matrix(rot_mat_t).as_euler("xyz")
            
        return pos, euler, quat

    def _handle_calibration(self, pose, joy_press, joy_release):
        if joy_press:
            self.calibrating_fwd = True
            self.calib_start_pose = pose.copy()
            print(f"[{self.side}] Calibrating Forward... Move controller forward")
        elif joy_release and self.calibrating_fwd:
            self.calibrating_fwd = False
            start = self.calib_start_pose[:3, 3]
            end = pose[:3, 3]
            vec = end - start
            dist = np.linalg.norm(vec)
            
            if dist > 0.003:
                # Calculate rotation to align movement with Robot +X
                fwd = vec / dist
                # Transform vector to env frame to see where it points
                temp_mat = np.eye(4)
                temp_mat[:3, 3] = fwd
                t_fwd = (self.global_to_env_mat @ temp_mat)[:3, 3]
                
                robot_fwd = np.array([1.0, 0.0, 0.0])
                
                # Calculate Rotation Matrix R_calib
                axis = np.cross(t_fwd, robot_fwd)
                angle = np.arccos(np.clip(np.dot(t_fwd, robot_fwd), -1.0, 1.0))
                
                if np.linalg.norm(axis) > 1e-3:
                    axis = axis / np.linalg.norm(axis)
                    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                    R_calib = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle)) * (K@K)
                else:
                    R_calib = np.eye(3)
                    
                self.vr_to_global_mat = np.eye(4)
                self.vr_to_global_mat[:3, :3] = R_calib
                
                # Include starting orientation
                try:
                    self.vr_to_global_mat = np.linalg.inv(self.calib_start_pose) @ self.vr_to_global_mat
                except: pass
                
                self.vr_neutral_pose = pose.copy()
                print(f"[{self.side}] ? Forward calibrated (Dist: {dist*1000:.1f}mm)")
                self.reset_to_home() # Reset robot after calib
            else:
                # Fallback: Just reset orientation at current pose
                self.vr_neutral_pose = pose.copy()
                # Basic Inverse
                try:
                    self.vr_to_global_mat = np.linalg.inv(pose)
                except: 
                    self.vr_to_global_mat = np.eye(4)
                
                print(f"[{self.side}] Orientation Reset (No movement).")
                # self.reset_to_home()

# [前文 ArmController 类保持不变...]

# --- Main Server Class (Dual-Arm Supported) ---

class OculusVRServer:
    def __init__(self, 
                 debug=False, 
                 mode='single', # 'single' or 'dual'
                 left_ip=None,
                 right_ip=None,
                 coord_transform=None,
                 performance_mode=True,
                 enable_recording=True,
                 camera_configs=None,
                 verify_data=False,
                 camera_config_path=None,
                 enable_cameras=True,
                 **kwargs):
        
        self.debug = debug
        self.mode = mode
        self.running = True
        self.verify_data = verify_data
        self.enable_recording = enable_recording
        
        # --- Robot Controllers (Dual Arm Setup) ---
        # 使用字典存储一个或两个臂
        self.arms = {}
        
        # 配置机械臂 IP
        # 如果没有传入特定 IP，使用默认值
        default_l_ip = "192.168.1.24" # 示例默认IP
        default_r_ip = "192.168.1.25"
        
        l_ip = left_ip if left_ip else default_l_ip
        r_ip = right_ip if right_ip else default_r_ip
        
        # 初始化机械臂
        if self.mode == 'dual':
            print("? Initializing Dual-Arm Mode...")
            self.arms['left'] = ArmController('left', l_ip, coord_transform, debug, performance_mode)
            self.arms['right'] = ArmController('right', r_ip, coord_transform, debug, performance_mode)
            self.main_controller_id = 'r' # 右手柄为主控
        else:
            # 单臂模式 (默认使用传入的 right_ip 或参数决定的单臂)
            # 这里为了兼容原来的 right_controller 参数逻辑，默认初始化右臂，除非特定需求
            print("? Initializing Single-Arm Mode (Right)...")
            self.arms['right'] = ArmController('right', r_ip, coord_transform, debug, performance_mode)
            self.main_controller_id = 'r'

        self.control_hz = CONTROL_FREQ * (2 if performance_mode else 1)
        self.control_interval = 1.0 / self.control_hz
        self.recording_hz = self.control_hz
        
        # --- Oculus Reader ---
        print("? Initializing Oculus Reader...")
        try:
            self.oculus_reader = OculusReader(ip_address=None, print_FPS=False) # ip_address=None for USB
            print("? Oculus Reader initialized successfully")
        except Exception as e:
            print(f"? Failed to initialize Oculus Reader: {e}")
            sys.exit(1)

        # --- Data Recording & Cameras ---
        self.data_recorder = None
        self.recording_active = False
        self.camera_manager = None
        
        if enable_cameras and camera_config_path:
            try:
                #  First, test cameras before starting the manager
                print("\n? Testing camera functionality...")
                from Modules.camera_test import test_cameras
              
                # Load camera config
                import yaml
                with open(self.camera_config_path, 'r') as f:
                    test_camera_configs = yaml.safe_load(f)
                
                # Run camera tests
                all_passed, test_results = test_cameras(test_camera_configs)
                
                if not all_passed:
                    print("\n? Camera tests failed!")
                    print("   Some cameras are not functioning properly.")
                    response = input("\n   Continue anyway? (y/N): ")
                    if response.lower() != 'y':
                        print("   Exiting due to camera test failures.")
                        sys.exit(1)
                    else:
                        print("   Continuing with available cameras...")
                        
                        
                        
                        
                from Modules.camera_manager import CameraManager
                self.camera_manager = CameraManager(camera_config_path)
                print("? Camera manager initialized")
            except Exception as e:
                print(f"??  Failed to initialize camera manager: {e}")
        
        if self.enable_recording:
            self.data_recorder = MCAPDataRecorder(
                camera_configs=camera_configs,
                save_images=True,
                save_depth=True,
                camera_manager=self.camera_manager
            )
            print("? MCAP data recording enabled")
            
        # --- Threading & Queues ---
        self.mcap_queue = queue.Queue(maxsize=1000)
        self._threads = []
        self._vr_state_lock = threading.Lock()
        self._latest_vr_state = None
        
        # Logic Flags
        self.prev_a_button = False
        # 动作缓存，用于录制。双臂模式下是 14维，单臂 7维
        action_dim = 14 if self.mode == 'dual' else 7
        self._last_action = np.zeros(action_dim)
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self._start_threads()
        
        print("\n? Oculus VR Server with DROID-exact VRPolicy Control")
        print(f"   Control frequency: {self.control_hz}Hz")
        print("\n? Controls:")
        print("   - HOLD grip button: Enable teleoperation")
        print("   - RELEASE grip button: Pause teleoperation")
        print("   - PRESS trigger: Close gripper")
        print("   - RELEASE trigger: Open gripper")
        if self.enable_recording:
            print("\n? Recording Controls:")
            print("   - A button: Start recording or stop current recording")
            print("   - B button: Mark recording as successful and save")
            print("   - Recordings saved to: ~/recordings/success")
            print("   - Stopped recordings (via A button) are discarded")
        else:
            print("   - A/X button: Mark success and exit")
            print("   - B/Y button: Mark failure and exit")
        print("\n? Forward Direction Calibration:")
        print("   - HOLD joystick button and MOVE controller forward")
        print("   - The direction you move defines 'forward' for the robot")
        print("   - Move at least 3mm in your desired forward direction")
        print("   - Release joystick button to complete calibration")
        
        print("\n? Tips:")
        print("   - Calibrate forward direction before starting teleoperation")
        print("   - Each grip press/release recalibrates the origin (starting position)")
        print("   - Robot movements are relative to the position when grip was pressed")
        
    
        
        print("\n??  Note: Using Deoxys control interface (quaternion-based)")
        print("   Rotations are handled differently than DROID's Polymetis interface")
        
        print("\n? Hot Reload:")
        print("   - Run with --hot-reload flag to enable automatic restart on code changes")
        print("   - Use: ./run_server.sh --hot-reload")
        print("   - The server will restart automatically when you save changes")
        
        print("\nPress Ctrl+C to exit gracefully\n")
    def _start_threads(self):
        self._state_thread = threading.Thread(target=self._vr_polling_worker)
        self._state_thread.daemon = True
        self._state_thread.start()
        
        if self.enable_recording:
            self._mcap_thread = threading.Thread(target=self._mcap_writer_worker)
            self._mcap_thread.daemon = True
            self._mcap_thread.start()
            self._threads.append(self._mcap_thread)
            
            self._record_thread = threading.Thread(target=self._data_recording_worker)
            self._record_thread.daemon = True
            self._record_thread.start()
            self._threads.append(self._record_thread)

    def signal_handler(self, signum, frame):
        print(f"\n? Received signal {signum}, shutting down...")
        self.stop_server()

    def _vr_polling_worker(self):
        while self.running:
            time.sleep(0.02)
            try:
                poses, buttons = self.oculus_reader.get_transformations_and_buttons()
                if not poses: continue
                
                # Check grip on MASTER controller for movement enabled flag
                # 在双臂模式下，通常只有当两个手柄都就位或者主手柄按下时才视为 Enabled
                # 这里简化逻辑：只要主手柄(右)按下侧键，就允许录制状态标记为 Enabled
                grip = buttons.get(self.main_controller_id.upper() + "G", False)
                
                vr_state = VRState(
                    timestamp=time.time(),
                    poses=copy.deepcopy(poses),
                    buttons=copy.deepcopy(buttons),
                    movement_enabled=grip,
                    controller_on=True
                )
                
                with self._vr_state_lock:
                    self._latest_vr_state = vr_state
                    
            except Exception as e:
                print(f"VR Poll Error: {e}")

    def control_loop(self):
        print(f"? Control Loop Started ({self.control_hz} Hz) | Mode: {self.mode.upper()}")
        if self.camera_manager: self.camera_manager.start()
        
        last_time = time.time()
        
        while self.running:
            try:
                now = time.time()
                if now - last_time < self.control_interval:
                    time.sleep(0.001)
                    continue
                last_time = now
                
                # 1. Update Robot States (Both arms)
                all_robots_ready = True
                for arm in self.arms.values():
                    state = arm.update_state()
                    if state is None: all_robots_ready = False
                
                # 2. Get VR State
                with self._vr_state_lock:
                    vr_state = self._latest_vr_state
                
                if vr_state and all_robots_ready:
                    current_actions = []
                    
                    # 3. Process Control for EACH arm
                    # 顺序很重要：如果是双臂，我们通常约定先左后右，或者先右后左
                    # 为了数据对其，我们固定顺序：Left, Right (如果存在)
                    order = ['left', 'right'] if self.mode == 'dual' else ['right']
                    
                    for side in order:
                        if side in self.arms:
                            action, info = self.arms[side].process_control(vr_state)
                            current_actions.append(action)
                    
                    # 拼接动作向量 (7 + 7 = 14) 或 (7)
                    self._last_action = np.concatenate(current_actions)
                    
                    # 4. Handle Recording Buttons (Only check Main Controller)
                    self._handle_recording_logic(vr_state.buttons)
                
            except Exception as e:
                print(f"Control Loop Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

    def _handle_recording_logic(self, buttons):
        if not self.enable_recording: 
            if buttons.get("A", False) or buttons.get("X", False):
                self.stop_server()
            return

        # Always use Right Controller (A/B) for system commands to avoid confusion
        btn_start = buttons.get("A", False)
        btn_save = buttons.get("B", False)

        if btn_start and not self.prev_a_button:
            if self.recording_active:
                print("\n? Stop Recording (Discard)")
                self.data_recorder.reset_recording()
                self.recording_active = False
            else:
                print("\n?? Start Recording")
                self.data_recorder.start_recording()
                self.recording_active = True
        self.prev_a_button = btn_start
        
        if btn_save and self.recording_active:
            print("\n? Save Recording")
            path = self.data_recorder.stop_recording(success=True)
            self.recording_active = False
            if self.verify_data and path:
                try: MCAPVerifier(path).verify(verbose=True)
                except: pass

    def _data_recording_worker(self):
        """Snapshot data at recording frequency"""
        last_time = time.time()
        while self.running:
            now = time.time()
            if now - last_time < (1.0/self.recording_hz):
                time.sleep(0.001)
                continue
            last_time = now
            
            if self.recording_active and self.data_recorder:
                with self._vr_state_lock:
                    vr = self._latest_vr_state
                
                # Check if all arms have valid state
                ready = True
                combined_robot_state = {} # We need to construct a composite state object or dict
                
                # Snapshot current states
                robot_states_list = []
                order = ['left', 'right'] if self.mode == 'dual' else ['right']
                
                for side in order:
                    if side in self.arms:
                        s = self.arms[side].current_state
                        if s is None: 
                            ready = False
                            break
                        robot_states_list.append(s)
                #组合数据
                if vr and ready:
                    # Construct composite robot state for the TimestepData
                    # We create a dummy structure that holds the concatenated data
                    # This keeps the MCAP writer logic cleaner
                    
                    # Concatenate arrays
                    cat_pos = np.concatenate([s.pos for s in robot_states_list])
                    cat_euler = np.concatenate([s.euler for s in robot_states_list])
                    cat_joints = np.concatenate([s.joint_positions for s in robot_states_list])
                    # Gripper is scalar, make it vector
                    cat_gripper = np.array([s.gripper for s in robot_states_list]) 

                    # Hack: create a pseudo RobotState object that holds combined data
                    # timestamp uses the first arm's timestamp
                    composite_state = RobotState(
                        timestamp=robot_states_list[0].timestamp,
                        pos=cat_pos,
                        quat=None, # Quat concatenation is messy, usually not used in flat raw state
                        euler=cat_euler,
                        gripper=cat_gripper, 
                        joint_positions=cat_joints
                    )

                    timestep = TimestepData(
                        timestamp=now,
                        vr_state=vr,
                        robot_state=composite_state, 
                        action=self._last_action.copy(),
                        info={"movement_enabled": vr.movement_enabled}
                    )
                    try:
                        self.mcap_queue.put_nowait(timestep)
                    except queue.Full: pass

    def _mcap_writer_worker(self):
        while self.running:
            try:
                data = self.mcap_queue.get(timeout=0.5)
                
                # Format for Labelbox/DROID
                # If dual arm, these arrays are already concatenated in _data_recording_worker
                
                # Handle Gripper formatting (scalar vs array)
                grip_data = data.robot_state.gripper
                if isinstance(grip_data, np.ndarray):
                    grip_list = grip_data.tolist()
                else:
                    grip_list = [grip_data]

                timestep_dict = {
                    "observation": {
                        "timestamp": {
                            "robot_state": {"read_start": int(data.timestamp*1e9), "read_end": int(data.timestamp*1e9)}
                        },
                        "robot_state": {
                            "cartesian_position": np.concatenate([data.robot_state.pos, data.robot_state.euler]).tolist(),
                            "gripper_position": grip_list,
                            "joint_positions": data.robot_state.joint_positions.tolist()
                        },
                        "controller_info": data.info
                    },
                    "action": data.action.tolist()
                }
                self.data_recorder.write_timestep(timestep_dict, data.timestamp)
            except queue.Empty: continue
            except Exception as e: print(f"MCAP Write Error: {e}")

    def start(self):
        try:
            self.control_loop()
        except KeyboardInterrupt:
            self.stop_server()

    def stop_server(self):
        if not self.running: return
        print("Stopping Server...")
        self.running = False
        if self.recording_active and self.data_recorder:
            self.data_recorder.stop_recording(success=False)
        if self.camera_manager: self.camera_manager.stop()
        if self.oculus_reader: self.oculus_reader.stop()
        for arm in self.arms.values():
            arm.disconnect()
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Oculus VR Server for JAKA (Dual/Single)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    # Removed --left-controller flag as it's superseded by --dual-arm or explicit single config
    parser.add_argument('--dual-arm', action='store_true', help='Enable Dual Arm Mode')
    parser.add_argument('--left-ip', type=str, default="192.168.1.24", help='IP for Left Robot')
    parser.add_argument('--right-ip', type=str, default="192.168.1.25", help='IP for Right Robot')
    parser.add_argument('--ip', type=str, default=None, help='IP address of Quest')
    parser.add_argument('--coord-transform', nargs='+', type=float, help='Custom transform')
    parser.add_argument('--performance', action='store_true', help='Enable performance mode')
    parser.add_argument('--no-recording', action='store_true', help='Disable recording')
    parser.add_argument('--enable-cameras', action='store_true', help='Enable cameras')
    
    args = parser.parse_args()
    
    mode = 'dual' if args.dual_arm else 'single'
    
    server = OculusVRServer(
        debug=args.debug,
        mode=mode,
        left_ip=args.left_ip,
        right_ip=args.right_ip,
        ip_address=args.ip,
        coord_transform=args.coord_transform,
        performance_mode=args.performance,
        enable_recording=not args.no_recording,
        camera_config_path='configs/cameras_intel.yaml',
        enable_cameras=args.enable_cameras
    )
    
    server.start()

if __name__ == "__main__":
    main()