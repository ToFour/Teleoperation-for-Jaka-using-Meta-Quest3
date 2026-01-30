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


'''
代码重构思路  为了双臂的拓展
重构后

oculus VRserver 负责VR连接，机器人控制，数据录制，和位姿映射
新创建一个arm controller负责单臂的所有控制逻辑
oculusVRserver 只负责VR连接，数据录制，和位姿映射

'''
@dataclass
class VRState:
    """Thread-safe VR controller state"""
    timestamp: float
    poses: Dict
    buttons: Dict
    left_movement_enabled: bool
    right_movement_enabled: bool
    controller_on: bool
    left_grip_released: bool
    right_grip_released: bool
    left_toggle_state: bool
    right_toggle_state: bool
    def copy(self):
        """Deep copy for thread safety"""
        return VRState(
            timestamp=self.timestamp,
            poses=copy.deepcopy(self.poses),
            buttons=copy.deepcopy(self.buttons),
            left_movement_enabled=self.left_movement_enabled,
            right_movement_enabled=self.right_movement_enabled,
            controller_on=self.controller_on,
            left_toggle_state=self.left_toggle_state,
            right_toggle_state=self.right_toggle_state,
            left_toggle_state=self.left_toggle_state,
            right_toggle_state=self.right_toggle_state
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
    def __init__(self, side: str, ip_address: str, coord_transform=None, debug=False, performance=True):
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
        # self._init_robot()

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
    def calculate_action(self, vr_pos, vr_quat, vr_euler, vr_gripper,
                                               robot_pos, robot_quat, robot_euler, robot_gripper,
                                               vr_origin_pos=None, vr_origin_quat=None, vr_origin_euler=None,
                                               robot_origin_pos=None, robot_origin_quat=None, robot_origin_euler=None):
        """
        Calculate robot action from VR and robot states
        输入: VR状态(pos, quat, euler, gripper) 和 机器人状态(pos, quat, euler, gripper)
        输出: action (7维: [linear_vel_x, linear_vel_y, linear_vel_z, rot_vel_x, rot_vel_y, rot_vel_z, gripper_vel])
        """
        # 如果没有提供原点，则使用当前状态作为原点
        if vr_origin_pos is None:
            vr_origin_pos = vr_pos
        if vr_origin_quat is None:
            vr_origin_quat = vr_quat
        if vr_origin_euler is None:
            vr_origin_euler = vr_euler
        if robot_origin_pos is None:
            robot_origin_pos = robot_pos
        if robot_origin_quat is None:
            robot_origin_quat = robot_quat
        if robot_origin_euler is None:
            robot_origin_euler = robot_euler

        # Calculate Positional Action - DROID exact
        robot_pos_offset = robot_pos - robot_origin_pos
        target_pos_offset = vr_pos - vr_origin_pos
        pos_action = target_pos_offset - robot_pos_offset

        # Calculate Rotation Action
        # VR controller's relative rotation from its origin
   

        # Calculate target euler offset
        robot_euler_offset = compute_angle_increment(robot_euler[:3], robot_origin_euler[:3])
        target_euler_offset = compute_angle_increment(vr_euler[:3], vr_origin_euler[:3])
        euler_action_final = np.array(target_euler_offset) - np.array(robot_euler_offset)

        

        # Apply deadzones
        for i in range(3):
            if abs(pos_action[i]) < self.output_deadzone:
                pos_action[i] = 0.0
        for i in range(3):
            if abs(euler_action_final[i]) < self.output_euler_deadzone:
                euler_action_final[i] = 0.0

        # Scale appropriately - DROID exact gains
        pos_action *= self.pos_gain
        euler_action_final *= self.rot_gain
        gripper_action *= self.gripper_gain
        
        # Apply velocity limits
        lin_vel_norm = np.linalg.norm(pos_action)
        rot_vel_norm = np.linalg.norm(euler_action_final)
     
        
        if lin_vel_norm > self.max_lin_vel:
            pos_action *= self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            euler_action_final *= self.max_rot_vel / rot_vel_norm
        
        # Prepare return values
        action = np.concatenate([pos_action, euler_action_final, [0]])
        action = action.clip(-1, 1)
        
        return action
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


class OculusVRServer:
    def __init__(self, 
                 debug=False, 
                 right_controller=True, 
                 ip_address=None,
                 simulation=False,
                 coord_transform=None,
                 rotation_mode="labelbox",
                 performance_mode=False,
                 enable_recording=True,
                 camera_configs=None,
                 verify_data=False,
                 camera_config_path=None,
                 enable_cameras=True):
        """
        Initialize the Oculus VR Server with DROID-exact VRPolicy control
        
        Args:
            debug: If True, only print data without controlling robot
            right_controller: If True, use right controller for robot control
            ip_address: IP address of Quest device (None for USB connection)
            simulation: If True, use simulated FR3 robot instead of real hardware
            coord_transform: Custom coordinate transformation vector (default: adjusted for compatibility)
            rotation_mode: Rotation mapping mode - currently only "labelbox" is supported
            performance_mode: If True, enable performance optimizations
            enable_recording: If True, enable MCAP data recording functionality
            camera_configs: Camera configuration dictionary for recording
            verify_data: If True, verify MCAP data after successful recording
            camera_config_path: Path to camera configuration JSON file
            enable_cameras: If True, enable camera recording
        """
        self.debug = debug
        self.right_controller = right_controller
        self.simulation = simulation
        self.running = True
        self.verify_data = verify_data
        
        #Robot config
        self._robot_ip="192.168.1.25"
        self._robot=JakaRobot("192.168.1.25")
        self.left_arm = ArmController("left", "192.168.1.24")
        self.right_arm = ArmController("right", "192.168.1.25")
        # self._robot=None
        self._robot_speed = 10
        self._robot_acc = 10
        self._robot_initpos = SAFE_POSITION
        self._robot_lastgripperState_left=GRIPPER_OPEN
        self._robot_gripper_count_left = 0
        self._robot_lastgripperState_right=GRIPPER_OPEN
        self._robot_gripper_count_right = 0
        self.arm_states={
            'left':RobotState(timestamp=0,pos=None,quat=None,euler=None,gripper=0.0,joint_positions=None),
            'right':RobotState(timestamp=0,pos=None,quat=None,euler=None,gripper=0.0,joint_positions=None)
        }
        
        
        # DROID VRPolicy exact parameters - no customization
        self.max_lin_vel = 1.0
        self.max_rot_vel = 1.0
        self.max_gripper_vel = 1.0
        self.spatial_coeff = 1.0
        self.pos_action_gain = 5.0
        self.rot_action_gain = 2.0
        self.gripper_action_gain = 3.0
        self.control_hz = CONTROL_FREQ  # Use constant from config
        # self.control_hz = 70 # Use constant from config
        self.control_interval = 1.0 / self.control_hz
        
        # DROID IK solver parameters for velocity-to-delta conversion
        self.max_lin_delta = 0.004  # Maximum linear movement per control cycle
        self.max_rot_delta =0.007    # Maximum rotation per control cycle (radians)
        self.max_gripper_delta = 0.25  # Maximum gripper movement per control cycle
        
        # Coordinate transformation
        # Default: DROID exact from oculus_controller.py: rmat_reorder: list = [-2, -1, -3, 4]
        # But this might not be correct for all robot setups
        #左右手柄可能不一样
        if coord_transform is None:
            # Try a more standard transformation that should work for most robots
            # This maps: VR +Z (forward) → Robot +X (forward)
            #           VR +X (right) → Robot -Y (right) 
            #           VR +Y (up) → Robot +Z (up)
            #VR [z x y] robot [x y z  ]
            # rmat_reorder = [-3, -1, 2, 4]  # More intuitive default  当没进行校准时，这里能进行初步的坐标轴映射
            rmat_reorder = [2, 1, -3, 4] 
            rmat_reorder_left = [2, 1, -3, 4] 
            rmat_reorder_right = [2, 1, -3, 4] 
            print("\n??  Using adjusted coordinate transformation for better compatibility")
            print("   If rotation is still incorrect, try --coord-transform with different values")
        else:
            rmat_reorder = coord_transform
            
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder) # 全局坐标转换矩阵
        self.global_to_env_mat_left = vec_to_reorder_mat(rmat_reorder_left) # 全局坐标转换矩阵
        self.global_to_env_mat_right = vec_to_reorder_mat(rmat_reorder_right) # 全局坐标转换矩阵
        # Rotation transformation matrix (separate from position)
        self.rotation_mode = rotation_mode
        # For labelbox mode, we apply euler angle transformations directly in _process_reading
        # to avoid creating invalid rotation matrices
        
        # if self.debug or coord_transform is not None:
        #     print("\n? Coordinate Transformation:")
        #     print(f"   Position reorder vector: {rmat_reorder}")
        #     print(f"   Rotation mode: {rotation_mode}")
        #     print("   Position Transformation Matrix:")
        #     for i in range(4):
        #         row = self.global_to_env_mat[i]
        #         print(f"   [{row[0]:6.1f}, {row[1]:6.1f}, {row[2]:6.1f}, {row[3]:6.1f}]")
            
        #     # Show what this transformation does
        #     test_vecs = [
        #         ([1, 0, 0, 1], "VR right (+X)"),
        #         ([0, 1, 0, 1], "VR up (+Y)"),
        #         ([0, 0, 1, 1], "VR forward (+Z)"),
        #     ]
        #     print("\n   Position Mapping:")
        #     for vec, name in test_vecs:
        #         transformed = self.global_to_env_mat @ np.array(vec)
        #         direction = ""
        #         if abs(transformed[0]) > 0.5:
        #             direction = f"Robot {'forward' if transformed[0] > 0 else 'backward'} (X)"
        #         elif abs(transformed[1]) > 0.5:
        #             direction = f"Robot {'left' if transformed[1] > 0 else 'right'} (Y)"
        #         elif abs(transformed[2]) > 0.5:
        #             direction = f"Robot {'up' if transformed[2] > 0 else 'down'} (Z)"
        #         print(f"   {name} → {direction}")
            
        #     # Show rotation mapping
        #     print("\n   Rotation Mapping (Labelbox mode):")
        #     print("   VR Roll → Robot Roll (INVERTED for ergonomics)")
        #     print("   VR Pitch → Robot Pitch")
        #     print("   VR Yaw → Robot Yaw")
        #     print("   Axis transform: [X, Y, Z] → [-Y, X, Z]")
        
        # Initialize transformation matrices
        self.vr_to_global_mat = np.eye(4)
        self.vr_to_global_mat_left = np.eye(4)
        self.vr_to_global_mat_right = np.eye(4)
        
        # Controller ID
        self.controller_id = "r" if right_controller else "l"
        
        # Initialize state
        self.reset_state()
        
        # Initialize Oculus Reader
        print("? Initializing Oculus Reader...")
        try:
            self.oculus_reader = OculusReader(
                ip_address=ip_address,
                print_FPS=False
            )
            print("? Oculus Reader initialized successfully")
        except Exception as e:
            print(f"? Failed to initialize Oculus Reader: {e}")
            sys.exit(1)

        
        #这里换成封装的初始化函数
        if not self.debug:                    
            self.left_arm._init_robot()
            self.right_arm._init_robot()      
            
       
        # Initialize MCAP data recorder
        self.enable_recording = enable_recording
        self.data_recorder = None
        self.recording_active = False
        self.prev_a_button = False  # For edge detection
        
        # Initialize camera manager
        self.camera_manager = None
        self.enable_cameras = enable_cameras
        self.camera_config_path = camera_config_path
        
        if self.enable_cameras and self.camera_config_path:
            try:
                # First, test cameras before starting the manager
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
                
                # Initialize camera manager after tests pass
                from Modules.camera_manager import CameraManager
                self.camera_manager = CameraManager(self.camera_config_path)
                print("? Camera manager initialized")
            except Exception as e:
                print(f"??  Failed to initialize camera manager: {e}")
                self.camera_manager = None
        
        if self.enable_recording:
            self.data_recorder = MCAPDataRecorder(
                camera_configs=camera_configs,
                save_images=True,
                save_depth=True,  # Enable depth recording if cameras support it
                camera_manager=self.camera_manager
            )
            print("? MCAP data recording enabled")
       
       
       
       
        # Setup signal handlers for graceful shutdown？
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Start state listening thread
        self._state_thread = threading.Thread(target=self._update_internal_state)
        self._state_thread.daemon = True
        self._state_thread.start()
        
        print("\n? Oculus VR Server with DROID-exact VRPolicy Control")
        print(f"   Using {'RIGHT' if right_controller else 'LEFT'} controller")
        print(f"   Mode: {'DEBUG' if debug else 'LIVE ROBOT CONTROL'}")
        print(f"   Robot: {'SIMULATED FR3' if simulation else 'REAL HARDWARE'}")
        print(f"   Control frequency: {self.control_hz}Hz")
        print(f"   Position gain: {self.pos_action_gain}")
        print(f"   Rotation gain: {self.rot_action_gain}")
        print(f"   Gripper gain: {self.gripper_action_gain}")
        
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
        
        # Performance tuning parameters (can be adjusted for responsiveness)
        self.enable_performance_mode = performance_mode  # Enable performance optimizations
        
        if self.enable_performance_mode:
            # Increase control frequency for tighter tracking
            self.control_hz = CONTROL_FREQ * 2  # Double the frequency for faster response
            self.control_interval = 1.0 / self.control_hz
            
            # Increase gains for more aggressive tracking
            #kp
            self.pos_action_gain = 18.0  # Even higher for tighter translation
            self.rot_action_gain = 1.5# Increased from 2.0
            
            # Adjust max deltas for higher frequency
            # Since we're running at 2x frequency, we can use smaller deltas per cycle
            # but achieve the same or better overall speed
            #调整每个cycle 的delta大小
            
            self.max_lin_delta = 0.005 # Maximum linear movement per control cycle
            self.max_rot_delta =0.008 
            print("\n? PERFORMANCE MODE ENABLED:")
            print(f"   Control frequency: {self.control_hz}Hz (2x faster)")
            print(f"   Position gain: {self.pos_action_gain} (100% higher)")
            print(f"   Rotation gain: {self.rot_action_gain} (50% higher)")
            print(f"   Tighter tracking for better translation following")
        self.output_deadzone = 0.003
        self.output_euler_deadzone = radians(5)
        
        # Translation tracking improvements
        #vr数据的处理
        self.translation_deadzone = 0.005  # 0.5mm deadzone to filter noise
        self.use_position_filter = True  # Filter out noise in position tracking
        self.position_filter_alpha = 0.9  # Higher = more responsive, lower = more smooth
        self._last_vr_pos = None  # For position filtering
        
        # Additional DROID IK solver parameters for enhanced control
        #可能没用？
        self.relative_max_joint_delta = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.max_joint_delta = self.relative_max_joint_delta.max()
        self.nullspace_gain = 0.025  # Gain for nullspace control (joint centering)
        self.regularization_weight = 1e-2  # Regularization for IK solver stability
        self.enable_joint_position_limits = True
        self.minimum_distance_from_joint_position_limit = 0.3  # radians
        self.joint_position_limit_velocity_scale = 0.95  # Scale velocity near limits
        self.max_cartesian_velocity_control_iterations = 300
        self.max_nullspace_control_iterations = 300
        
        # Asynchronous components
        self._vr_state_lock = threading.Lock()
        self._robot_state_lock = threading.Lock()
        self._robot_comm_lock = threading.Lock()  # Lock for robot communication?
        self._latest_vr_state = None
        self._latest_robot_state = None
        
        # Thread-safe queues for data flow
        self.mcap_queue = queue.Queue(maxsize=1000)  # Buffer for MCAP writer
        self.control_queue = queue.Queue(maxsize=10)  # Commands to robot
        
        # Thread management
        self._threads = []
        self._mcap_writer_thread = None
        self._robot_control_thread = None
        self._control_paused = False  # Flag to pause control during reset
        
        # Recording at independent frequency
        self.recording_hz = self.control_hz  # Record at same rate as control target
        self.recording_interval = 1.0 / self.recording_hz
        
        # Async robot communication
        self._robot_command_queue = queue.Queue(maxsize=2)  # Small buffer for commands
        self._robot_response_queue = queue.Queue(maxsize=2)  # Responses from robot
        self._robot_comm_thread = None
        
    def reset_state(self):
        """Reset internal state - matches DROID exactly"""
        #同时运行两个手柄可能要有两个状态量
        self._state = {
            "poses": {},
            "buttons": {"A": False, "B": False, "X": False, "Y": False},
            "left_movement_enabled": False,
            "right_movement_enabled": False,
            "controller_on": True,
            "left_grip_released": False,
            "right_grip_released": False,
            "left_gripper_toggle_state": False,
            "right_gripper_toggle_state": False
        }
        self.update_sensor = True
        self.reset_origin_left = True
        self.reset_origin_right = True
        self.reset_orientation = True
        self.robot_origin_left = None
        self.robot_origin_right = None
        self.vr_origin_left = None
        self.vr_origin_right= None
        self.vr_state = None
        
        # Robot state - Deoxys uses quaternions directly
        #用于recorder的全局状态量，要改
        self.robot_pos = None
        self.robot_quat = None
        self.robot_euler = None
        self.robot_gripper = 0.0
        self.robot_joint_positions = None
        
        # Button state tracking for edge detection
        self.prev_joystick_state = False
        self.prev_grip_state = False
        
        # Calibration state
        self.calibrating_forward = False
        self.calibration_start_pose = None
        self.calibration_start_time = None
        self.vr_neutral_pose = None  # Neutral controller orientation
        self.vr_neutral_pose_left = None
        self.vr_neutral_pose_right = None
        # First frame flag
        self.is_first_frame = True
        
        # Flag to reset robot after calibration
        self._reset_robot_after_calibration = False
        
        # Added for rotation tracking
        self._last_controller_rot = None
        
        # Position filtering state
        self._last_vr_pos = None
        
        # Last action for recording thread
        self._last_action = np.zeros(14)
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C and other termination signals"""
        print(f"\n? Received signal {signum}, shutting down gracefully...")
        self.stop_server()
        
    def _update_internal_state(self, num_wait_sec=5, hz=50):
        """Continuously poll VR controller state at 50Hz - with intuitive calibration"""
        last_read_time = time.time()
        '''
        1.获取手柄位姿和按键状态
        2.获取当前grip（使能），trigger（夹爪）和joystick状态
        3.根据前一帧的状态计算grip的翻转和释放，joystick的按压（上升沿）与释放（下降沿），trigger的上升沿（夹爪）
        4.更新全局变量self._state和vrstate，prev_grip_state和prev_joystick_state
        
        '''
        prev_grip_state={"left":False,"right":False}
        prev_joystick_state={"left":False,"right":False}
        prev_trigger_state = {"left": 0.0, "right": 0.0}
        
        while self.running:
            # Regulate Read Frequency
            time.sleep(1 / hz)
            
            # Read Controller
            time_since_read = time.time() - last_read_time
            poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            self._state["controller_on"] = time_since_read < num_wait_sec
            
            if poses == {}:
                continue
            
            # Get current button states
            
            # #这里对左右区分了 
            # current_grip = buttons.get(self.controller_id.upper() + "G", False)
            # current_joystick = buttons.get(self.controller_id.upper() + "J", False)
            
            current_grip={  "left":buttons.get("LG", False),
                            "right" :  buttons.get("RG", False)           
                        }
            current_joystick = {    "left":buttons.get("LJ", False),
                                        "right" :  buttons.get("RJ", False)           
                                }
            current_trigger = {
                    "left": buttons.get("leftTrig", [0.0])[0],
                    "right": buttons.get("rightTrig", [0.0])[0]
                }
            # Detect edge transitions
            edge_events = {}
            for arm_side in ["left", "right"]:
                # Detect edge transitions for this arm
                grip_toggled = prev_grip_state[arm_side] != current_grip[arm_side]
                joystick_pressed = current_joystick[arm_side] and not prev_joystick_state[arm_side]  # Rising edge
                joystick_released = not current_joystick[arm_side] and prev_joystick_state[arm_side]  # Falling edge
                grip_released = not current_grip[arm_side] and prev_grip_state[arm_side]  # Falling edge
                
                # Trigger toggle logic
                trigger_pressed = current_trigger[arm_side] > 0.1
                was_trigger_pressed = prev_trigger_state[arm_side] > 0.1
                trigger_toggled = trigger_pressed and not was_trigger_pressed  # Rising edge
                
                edge_events[arm_side] = {
                    "grip_toggled": grip_toggled,
                    "joystick_pressed": joystick_pressed,
                    "joystick_released": joystick_released,
                    "grip_released": grip_released,
                    "trigger_toggled": trigger_toggled
                }  

          
            # Update control flags
            self.update_sensor = self.update_sensor or current_grip
            self.reset_origin = self.reset_origin or grip_toggled
            self.reset_origin_left= self.reset_origin_left or edge_events["left"]["grip_toggled"]
            self.reset_origin_right= self.reset_origin_right or edge_events["right"]["grip_toggled"]
            # Save Info
            self._state["poses"] = poses
            self._state["buttons"] = buttons
            self._state["left_movement_enabled"] = current_grip["left"]
            self._state["right_movement_enabled"] = current_grip["right"]
            self._state["controller_on"] = True
            #使能键释放标志位，用于停止机械臂运行
            if edge_events["left"]["grip_released"] and self.debug==False:
                print("   left Gripper released")
                self._state["left_grip_released"] = True#下降沿置位，等待回调释放
            if edge_events["right"]["grip_released"] and self.debug==False:
                print("   right Gripper released")
                self._state["right_grip_released"] = True#下降沿置位，等待回调释放
            #夹爪扳机上升沿触发
            if edge_events["left"]["trigger_toggled"] and self.debug==False:
                self._state["left_gripper_toggle_state"] = True#上升沿置位，等待回调释放
                self._robot_gripper_count_left+=1        
                if self._robot_gripper_count_left>1: self._robot_gripper_count_left=0
            if edge_events["right"]["trigger_toggled"] and self.debug==False:
                self._state["right_gripper_toggle_state"] = True#下降沿置位，等待回调释放
                self._robot_gripper_count_right+=1        
                if self._robot_gripper_count_right>1: self._robot_gripper_count_right=0
                
            last_read_time = time.time()
            
            # Publish VR state to async system
            current_time = time.time()
            vr_state = VRState(
                timestamp=current_time,
                poses=copy.deepcopy(poses),
                buttons=copy.deepcopy(buttons),
                left_movement_enabled=current_grip["left"],
                right_movement_enabled=current_grip["right"],
                controller_on=True,
                left_grip_released=edge_events["left"]["grip_released"],
                right_grip_released=edge_events["right"]["grip_released"],
                left_toggle_state=edge_events["left"]["trigger_toggled"],
                right_toggle_state=edge_events["right"]["trigger_toggled"]
            )
            
            with self._vr_state_lock:
                self._latest_vr_state = vr_state
            
            # Handle Forward Direction Calibration
            #校准定义机器人的正前方
            if self.controller_id in self._state["poses"]:
                pose_matrix = self._state["poses"][self.controller_id]
                
                # Start calibration when joystick is pressed
                if joystick_pressed:
                    self.calibrating_forward = True
                    self.calibration_start_pose = pose_matrix.copy()
                    self.calibration_start_time = time.time()
                    print(f"\n? Forward calibration started - Move controller in desired forward direction")
                    print(f"   Hold the joystick and move at least 3mm forward")
                
                # Complete calibration when joystick is released
                elif joystick_released and self.calibrating_forward:
                    self.calibrating_forward = False
                    
                    if self.calibration_start_pose is not None:
                        # Get movement vector
                        start_pos = self.calibration_start_pose[:3, 3]
                        end_pos = pose_matrix[:3, 3]
                        #计算前向向量
                        movement_vec = end_pos - start_pos
                        movement_distance = np.linalg.norm(movement_vec)
                        
                        if movement_distance > 0.003:  # 3mm threshold
                            # Normalize movement vector
                            forward_vec = movement_vec / movement_distance
                            
                            print(f"\n? Forward direction calibrated!")
                            print(f"   Movement distance: {movement_distance*1000:.1f}mm")
                            print(f"   Forward vector: [{forward_vec[0]:.3f}, {forward_vec[1]:.3f}, {forward_vec[2]:.3f}]")
                            
                            # Create rotation to align this vector with robot's forward
                            # First, apply the coordinate transform to see what this maps to
                            #建立临时矩阵
                            temp_mat = np.eye(4)
                            temp_mat[:3, 3] = forward_vec
                            transformed_temp = self.global_to_env_mat @ temp_mat
                            transformed_forward = transformed_temp[:3, 3]
                            
                            # Calculate rotation to align with robot's +X axis
                            robot_forward = np.array([1.0, 0.0, 0.0])
                            #叉积找到旋转轴
                            rotation_axis = np.cross(transformed_forward, robot_forward)
                            #点积找到角度
                            rotation_angle = np.arccos(np.clip(np.dot(transformed_forward, robot_forward), -1.0, 1.0))
                            
                            if np.linalg.norm(rotation_axis) > 0.001:
                                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                                # Create rotation matrix using Rodrigues' formula
                                K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                             [rotation_axis[2], 0, -rotation_axis[0]],
                                             [-rotation_axis[1], rotation_axis[0], 0]])
                                R_calibration = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * K @ K
                            else:
                                # Movement is already aligned with robot forward or backward
                                if transformed_forward[0] < 0:  # Moving backward
                                    R_calibration = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                                else:
                                    R_calibration = np.eye(3)
                            
                            # Update the VR to global transformation
                            self.vr_to_global_mat = np.eye(4)
                            self.vr_to_global_mat[:3, :3] = R_calibration
                            
                            # Also incorporate the starting orientation to match DROID behavior
                            # This makes the current controller orientation the "neutral" orientation
                            try:
                                self.vr_to_global_mat = np.linalg.inv(self.calibration_start_pose) @ self.vr_to_global_mat
                            except:
                                print("Warning: Could not invert calibration pose")
                            
                            self.reset_orientation = False  # Calibration complete
                            
                            # Store the current controller pose as the neutral reference
                            # Store the raw pose, not transformed
                            self.vr_neutral_pose = np.asarray(self._state["poses"][self.controller_id]).copy()
                            print("? Stored neutral controller orientation")
                            
                            # Reset robot to home position after calibration
                            if not self.debug and not self.reset_orientation:
                                print("? Moving robot to reset position after calibration...")
                                self._reset_robot_after_calibration = True
                            elif self.debug:
                                print("\n? Calibration complete! Ready for teleoperation.")
                                print("   Hold grip button to start controlling the robot")
                        else:
                            print(f"\n??  Not enough movement detected ({movement_distance*1000:.1f}mm)")
                            print(f"   Please move controller at least 3mm in your desired forward direction")
                            # Keep reset_orientation True to allow DROID-style calibration as fallback
                            self.reset_orientation = True
                
                # Show calibration progress
                elif self.calibrating_forward and current_joystick:
                    if time.time() - self.calibration_start_time > 0.5:  # Update every 0.5s
                        current_pos = pose_matrix[:3, 3]
                        start_pos = self.calibration_start_pose[:3, 3]
                        distance = np.linalg.norm(current_pos - start_pos) * 1000  # Convert to mm
                        print(f"   Current movement: {distance:.1f}mm", end='\r')
            
            # DROID-style calibration fallback (just pressing joystick without movement)
            if self.reset_orientation and not self.calibrating_forward:
                stop_updating = self._state["buttons"][self.controller_id.upper() + "J"] or self._state["left_movement_enabled"] or self._state["right_movement_enabled"]
                if stop_updating:
                    rot_mat = np.asarray(self._state["poses"][self.controller_id])
                    rot_mat_left=np.asarray(self._state["poses"]["l"])
                    rot_mat_right=np.asarray(self._state["poses"]["r"])
                    self.reset_orientation = False
                    # try to invert the rotation matrix, if not possible, then just use the identity matrix
                    try:
                        rot_mat = np.linalg.inv(rot_mat)
                        rot_mat_left = np.linalg.inv(rot_mat_left)
                        rot_mat_right = np.linalg.inv(rot_mat_right)
                    except:
                        print(f"exception for rot mat: {rot_mat}")
                        rot_mat = np.eye(4)
                        rot_mat_left = np.eye(4)
                        rot_mat_right = np.eye(4)
                        self.reset_orientation = True
                    self.vr_to_global_mat = rot_mat
                    self.vr_to_global_mat_left = rot_mat_left
                    self.vr_to_global_mat_right = rot_mat_right
                    print("? Orientation reset (DROID-style)")
                    
                    # Store the current controller pose as the neutral reference
                    # Store the raw pose, not transformed
                    self.vr_neutral_pose = np.asarray(self._state["poses"][self.controller_id]).copy()
                    self.vr_neutral_pose_left = np.asarray(self._state["poses"]["l"]).copy()
                    self.vr_neutral_pose_right = np.asarray(self._state["poses"]["r"]).copy()
                    print("? Stored neutral controller orientation")
                    
                    # # Reset robot to home position after calibration
                    # if not self.debug and not self.reset_orientation:
                    #     print("? Moving robot to reset position after calibration...")
                    #     self._reset_robot_after_calibration = True
            
            # Update previous button states for next iteration
            for arm_side in ["left", "right"]:
                prev_grip_state[arm_side] = current_grip[arm_side]
                prev_joystick_state[arm_side] = current_joystick[arm_side]
                prev_trigger_state[arm_side] = current_trigger[arm_side]
    
    def _process_reading(self):
        """Apply coordinate transformations to VR controller pose - DROID exact"""
        rot_mat = np.asarray(self._state["poses"][self.controller_id])
    
        rot_mat_left=np.asarray(self._state["poses"]["l"])
        rot_mat_right=np.asarray(self._state["poses"]["r"])
        # Apply position transformationself.global_to_env_mat_left 
        transformed_mat = self.global_to_env_mat @ self.vr_to_global_mat_left @ rot_mat
        transformed_mat_left =self.global_to_env_mat_left  @ self.vr_to_global_mat_left @ rot_mat_left
        transformed_mat_right = self.global_to_env_mat_right  @ self.vr_to_global_mat_right @ rot_mat
        #手柄位姿到机器人基底系的坐标=坐标轴映射@VR到基底系的转换矩阵 @ 当前手柄相对VR坐标系位姿
        # vr_pos = self.spatial_coeff * transformed_mat[:3, 3]#取齐次坐标
        vr_pos_left = self.spatial_coeff * transformed_mat_left[:3, 3]#取齐次坐标
        vr_pos_right = self.spatial_coeff * transformed_mat_right[:3, 3]#取齐次坐标
        vr_pos={
                "left":vr_pos_left,
                "right":vr_pos_right
        }
        vr_gripper={
                "left":None,
                "right":None
        }
        vr_quat={
                "left":None,
                "right":None
            }
        vr_euler={
                "left":None,
                "right":None
            }
        # Apply position filtering to reduce noise/drift
        if self.use_position_filter and self._last_vr_pos is not None:
            # Calculate position change
            #对增量进行死区滤波
            for arm_side in ["left", "right"]:
                pos_delta = vr_pos[arm_side] - self._last_vr_pos[arm_side]
                
                # Apply deadzone to filter out small movements (noise)
                for i in range(3):
                    if abs(pos_delta[i]) < self.translation_deadzone:
                        pos_delta[i] = 0.0
            
            # Update position with filtered delta
                vr_pos[arm_side] = self._last_vr_pos[arm_side] + pos_delta
            
            # Alternative: Exponential moving average filter
            # vr_pos = self.position_filter_alpha * vr_pos + (1 - self.position_filter_alpha) * self._last_vr_pos
        
        self._last_vr_pos = vr_pos.copy()
        
        # Handle rotation
        if hasattr(self, 'vr_neutral_pose') and (self.vr_neutral_pose_left  is not None):
            # Calculate relative rotation from neutral pose using quaternions
            neutral_rot_right = R.from_matrix(self.vr_neutral_pose_right[:3, :3])
            neutral_rot_left = R.from_matrix(self.vr_neutral_pose_left[:3, :3])
            
            current_rot_right= R.from_matrix(rot_mat_right[:3, :3])
            current_rot_left = R.from_matrix(rot_mat_left[:3, :3])
            
            # Get relative rotation as quaternion
            relative_rot_right = neutral_rot_right.inv() * current_rot_right
            relative_rot_left = neutral_rot_left.inv() * current_rot_left
            # Convert to axis-angle to apply transformation
            rotvec_right = relative_rot_right.as_rotvec()
            angle_right = np.linalg.norm(rotvec_right)
            
            rotvec_left = relative_rot_left.as_rotvec()
            angle_left = np.linalg.norm(rotvec_left)
           
            if angle_left > 0:
                axis = rotvec_left / angle_left  # Normalize to get unit axis
                
                # Transform the rotation axis according to labelbox mapping
                #姿态映射
                # VR axes (confirmed by calibration): X=pitch, Y=roll, Z=yaw
                # Robot axes: X=roll, Y=pitch, Z=yaw
                # Desired mapping:
                # VR Roll (Y-axis) → Robot Pitch (Y-axis) - INVERTED for ergonomics
                # VR Pitch (X-axis) → Robot Roll (X-axis)
                # VR Yaw (Z-axis) → Robot Yaw (Z-axis)
                # We need to swap X and Y components to achieve this
                # Negate Y for inverted roll
                transformed_axis = np.array([-axis[1], -axis[0], -axis[2]])
                
                # Create new rotation from transformed axis and same angle
                transformed_rotvec = transformed_axis * angle_left
                transformed_rot = R.from_rotvec(transformed_rotvec)
                vr_quat["left"]= transformed_rot.as_quat()
            else:
                # No rotation
                vr_quat["left"] = np.array([0, 0, 0, 1])
            if angle_right > 0:
                axis = rotvec_right / angle_right  # Normalize to get unit axis
                
                # Transform the rotation axis according to labelbox mapping
                #姿态映射
                # VR axes (confirmed by calibration): X=pitch, Y=roll, Z=yaw
                # Robot axes: X=roll, Y=pitch, Z=yaw
                # Desired mapping:
                # VR Roll (Y-axis) → Robot Pitch (Y-axis) - INVERTED for ergonomics
                # VR Pitch (X-axis) → Robot Roll (X-axis)
                # VR Yaw (Z-axis) → Robot Yaw (Z-axis)
                # We need to swap X and Y components to achieve this
                # Negate Y for inverted roll
                transformed_axis = np.array([-axis[1], -axis[0], -axis[2]])
                
                # Create new rotation from transformed axis and same angle
                transformed_rotvec = transformed_axis * angle_right
                transformed_rot = R.from_rotvec(transformed_rotvec)
                vr_quat["right"]= transformed_rot.as_quat()
            else:
                # No rotation
                vr_quat["right"] = np.array([0, 0, 0, 1])
        else:
            # No neutral pose yet, apply standard transformation
            transformed_rot_mat = self.global_to_env_mat[:3, :3] @ self.vr_to_global_mat[:3, :3] @ rot_mat[:3, :3]
            vr_quat["right"] = rmat_to_quat(transformed_rot_mat)
            vr_quat["left"] = rmat_to_quat(transformed_rot_mat)
        # vr_gripper = self._state["buttons"]["rightTrig" if self.controller_id == "r" else "leftTrig"][0]
        vr_gripper["right"] = self._state["buttons"]["rightTrig" ][0]
        vr_gripper["left"] = self._state["buttons"][ "leftTrig"][0]
        vr_euler["right"]= quat_to_euler(vr_quat["right"])
        vr_euler["left"]= quat_to_euler(vr_quat["left"])
        self.vr_state = {"pos": vr_pos, "quat": vr_quat, "gripper": vr_gripper, "euler": vr_euler}

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        """Scales down the linear and angular magnitudes of the action - DROID exact"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        gripper_vel_norm = np.linalg.norm(gripper_vel)
        
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        if gripper_vel_norm > self.max_gripper_vel:
            gripper_vel = gripper_vel * self.max_gripper_vel / gripper_vel_norm
        
        return lin_vel, rot_vel, gripper_vel
    
    def _calculate_action(self):
        """Calculate robot action from VR controller state - DROID exact implementation"""
        #input:self.robot_pos:[x,y,z],,euler,vr_pos,euler
        #output: lin_vel, rot_vel, gripper_vel
        # Read Sensor
        if self.update_sensor:
            self._process_reading()
            self.update_sensor = False
        
        # Check if we have valid data
        if self.vr_state is None or self.robot_pos is None:
            return np.zeros(14), {}
        
        # Reset Origin On Release
        if self.reset_origin:
            self.robot_origin = {"pos": self.robot_pos, "quat": self.robot_quat,"euler":self.robot_euler}
            self.vr_origin = {"pos": self.vr_state["pos"], "quat": self.vr_state["quat"],"euler":self.vr_state["euler"]}
            self.reset_origin = False
            print("? Origin calibrated")
            if self.debug:
                print(f"   Robot origin: pos=[{self.robot_origin['pos'][0]:.3f}, {self.robot_origin['pos'][1]:.3f}, {self.robot_origin['pos'][2]:.3f}]")
                euler = quat_to_euler(self.robot_origin['quat'], degrees=True)
                print(f"   Robot origin rotation: [R:{euler[0]:6.1f}, P:{euler[1]:6.1f}, Y:{euler[2]:6.1f}]")
        
        # Calculate Positional Action - DROID exact
        robot_pos_offset = self.robot_pos - self.robot_origin["pos"]
        target_pos_offset = self.vr_state["pos"] - self.vr_origin["pos"]
        pos_action = target_pos_offset - robot_pos_offset
        
        # robot_pos_offset = self.robot_pos - self.
        # target_pos_offset = self.vr_state["pos"] - self._last_vr_pos["pos"]
        # pos_action = target_pos_offset - robot_pos_offset
    
        # Debug translation tracking
        if self.debug and np.linalg.norm(pos_action) > 0.001:
            print(f"\n? Translation Debug:")
            print(f"   VR Origin: [{self.vr_origin['pos'][0]:.4f}, {self.vr_origin['pos'][1]:.4f}, {self.vr_origin['pos'][2]:.4f}]")
            print(f"   VR Current: [{self.vr_state['pos'][0]:.4f}, {self.vr_state['pos'][1]:.4f}, {self.vr_state['pos'][2]:.4f}]")
            print(f"   VR Offset: [{target_pos_offset[0]:.4f}, {target_pos_offset[1]:.4f}, {target_pos_offset[2]:.4f}]")
            print(f"   Robot Offset: [{robot_pos_offset[0]:.4f}, {robot_pos_offset[1]:.4f}, {robot_pos_offset[2]:.4f}]")
            print(f"   Pos Action: [{pos_action[0]:.4f}, {pos_action[1]:.4f}, {pos_action[2]:.4f}]")
            print(f"   Pos Action Norm: {np.linalg.norm(pos_action):.4f}")
        print(f"\n? Translation Debug:")
        print(f"   VR Origin: [{self.vr_origin['pos'][0]:.4f}, {self.vr_origin['pos'][1]:.4f}, {self.vr_origin['pos'][2]:.4f}]")
        print(f"   VR Current: [{self.vr_state['pos'][0]:.4f}, {self.vr_state['pos'][1]:.4f}, {self.vr_state['pos'][2]:.4f}]")
        print(f"   VR Offset: [{target_pos_offset[0]:.4f}, {target_pos_offset[1]:.4f}, {target_pos_offset[2]:.4f}]")
        print(f"   Robot Offset: [{robot_pos_offset[0]:.4f}, {robot_pos_offset[1]:.4f}, {robot_pos_offset[2]:.4f}]")
        print(f"   Pos Action: [{pos_action[0]:.4f}, {pos_action[1]:.4f}, {pos_action[2]:.4f}]")
        print(f"   Pos Action Norm: {np.linalg.norm(pos_action):.4f}")
        # Calculate Rotation Action - CRITICAL FIX FOR DEOXYS
        # DROID calculates euler velocities because Polymetis expects euler angles
        # But Deoxys expects quaternions directly, so we need a different approach
        
        # Calculate the target quaternion directly
        # VR controller's relative rotation from its origin
        vr_relative_rot = R.from_quat(self.vr_origin["quat"]).inv() * R.from_quat(self.vr_state["quat"])
        # Apply this relative rotation to the robot's origin orientation
        target_rot = R.from_quat(self.robot_origin["quat"]) * vr_relative_rot
        target_quat = target_rot.as_quat()
        
        # For velocity-based control, we still need to calculate the rotation difference
        # But we'll use it differently than DROID
        robot_quat_offset = quat_diff(self.robot_quat, self.robot_origin["quat"])
        target_quat_offset = quat_diff(self.vr_state["quat"], self.vr_origin["quat"])
        quat_action = quat_diff(target_quat_offset, robot_quat_offset)
        euler_action= quat_to_euler(quat_action)

        print(f"   Robot Quat Offset: [{robot_quat_offset[0]:.4f}, {robot_quat_offset[1]:.4f}, {robot_quat_offset[2]:.4f}, {robot_quat_offset[3]:.4f}]")
        print(f"   Target Quat Offset: [{target_quat_offset[0]:.4f}, {target_quat_offset[1]:.4f}, {target_quat_offset[2]:.4f}, {target_quat_offset[3]:.4f}]")
        print(f"   Quat Action: [{quat_action[0]:.4f}, {quat_action[1]:.4f}, {quat_action[2]:.4f}, {quat_action[3]:.4f}]")
        print(f"   Euler Action1: [{euler_action[0]:.4f}, {euler_action[1]:.4f}, {euler_action[2]:.4f}]")


        def compute_angle_increment(current_angles, reference_angles, is_radians=True):
            """     
            参数:
                current_angles: 当前角度数组
                reference_angles: 参考角度数组
                is_radians: 是否为弧度制，True表示弧度，False表示角度
                
            返回:
                增量角度数组
            """
            # 确保输入是NumPy数组
            current = np.asarray(current_angles, dtype=np.float64)
            reference = np.asarray(reference_angles, dtype=np.float64)
            
            if current.shape != reference.shape:
                raise ValueError(f"角度数组形状不一致: {current.shape} vs {reference.shape}")
            
            # 计算原始增量
            delta = current - reference
            
            if is_radians:
                two_pi = 2 * math.pi
                limit = math.pi
            else:
                two_pi = 360.0
                limit = 180.0
            
            # 向量化调整到(-limit, limit]范围
            # 使用mod运算更高效
            delta = (delta + limit) % two_pi - limit
            
            # 处理边界情况：确保-limit被映射到-limit而不是limit
            delta = np.where(delta <= -limit, delta + two_pi, delta)
            
            return delta



        # robot_euler_offset = self.robot_euler[:3]-self.robot_origin["euler"][:3]
        robot_euler_offset = compute_angle_increment(self.robot_euler[:3],self.robot_origin["euler"][:3])
        print(f"   Robot euler_ Offset11: [{np.degrees(robot_euler_offset[0]):.4f}, {np.degrees(robot_euler_offset[1]):.4f}, {np.degrees(robot_euler_offset[2]):.4f}]")
        target_euler_offset = compute_angle_increment(self.vr_state["euler"][:3], self.vr_origin["euler"][:3])
        euler_action = np.array(target_euler_offset) - np.array(robot_euler_offset)
        print(f"robot_euler:[{np.degrees(self.robot_euler[0]):.4f}, {np.degrees(self.robot_euler[1]):.4f}, {np.degrees(self.robot_euler[2]):.4f}]")
        print(f"robot_origin: [{np.degrees(self.robot_origin['euler'][0]):.4f}, {np.degrees(self.robot_origin['euler'][1]):.4f}, {np.degrees(self.robot_origin['euler'][2]):.4f}]")
        print(f"   Robot euler_ Offset: [{np.degrees(robot_euler_offset[0]):.4f}, {np.degrees(robot_euler_offset[1]):.4f}, {np.degrees(robot_euler_offset[2]):.4f}]")
        print(f"   Target euler_ Offset: [{np.degrees(target_euler_offset[0]):.4f}, {np.degrees(target_euler_offset[1]):.4f}, {np.degrees(target_euler_offset[2]):.4f}]")
        print(f"vr_state_euler: [{np.degrees(self.vr_state['euler'][0]):.4f}, {np.degrees(self.vr_state['euler'][1]):.4f}, {np.degrees(self.vr_state['euler'][2]):.4f}]")
        # print(f"   Euler Action2: [{euler_action2[0]:.4f}, {euler_action2[1]:.4f}, {euler_action2[2]:.4f}]")
        print(f"   Euler Action2: [{euler_action[0]}, {euler_action[1]}, {euler_action[2]}]")
        # Calculate Gripper Action - DROID exact
        gripper_action = (self.vr_state["gripper"] * 1.5) - self.robot_gripper
        
        # Calculate Desired Pose
        target_pos = pos_action + self.robot_pos
        target_euler = add_angles(euler_action, self.robot_euler)
        
        target_cartesian = np.concatenate([target_pos, target_euler])
        target_gripper = self.vr_state["gripper"]


        #死区
        for i in range(3):
            if abs(pos_action[i]) < self.output_deadzone:
                pos_action[i] = 0.0
        for i in range(3):
            if abs(euler_action[i]) < self.output_euler_deadzone:
                euler_action[i] = 0.0
        # Scale Appropriately - DROID exact gains
        print(f"\n? scale_euler_action Debug:{euler_action}")
        pos_action *= self.pos_action_gain
        euler_action *= self.rot_action_gain
        print(f"   scale_Pos Action: [{pos_action[0]:.4f}, {pos_action[1]:.4f}, {pos_action[2]:.4f}]")
        gripper_action *= self.gripper_action_gain
        
        # Apply velocity limits
        lin_vel, rot_vel, gripper_vel = self._limit_velocity(pos_action, euler_action, gripper_action)
        
        # Prepare Return Values
        # Store the target quaternion for Deoxys
        info_dict = {
            "target_cartesian_position": target_cartesian, 
            "target_gripper_position": target_gripper,
            "target_quaternion": target_quat  # Add this for Deoxys
        }
        action = np.concatenate([lin_vel, rot_vel, [gripper_vel]])
        action = action.clip(-1, 1)
        
        return action, info_dict
    
    def get_info(self):
        """Get controller state information - DROID exact"""
        info = {
            "success": self._state["buttons"]["A"] if self.controller_id == 'r' else self._state["buttons"]["X"],
            "failure": self._state["buttons"]["B"] if self.controller_id == 'r' else self._state["buttons"]["Y"],
            "left_movement_enabled": self._state["left_movement_enabled"],
            "right_movement_enabled": self._state["right_movement_enabled"],
            "controller_on": self._state["controller_on"],
        }
        
        # Add raw VR controller data for recording
        if self._state["poses"] and self._state["buttons"]:
            info["poses"] = self._state["poses"]
            info["buttons"] = self._state["buttons"]
        
        return info
        
    def reset_robot(self, sync=True):
        """Reset robot to initial position
        
       
        """
        if self.debug:
            print("? [DEBUG] Would reset robot to initial position")
            # Return simulated values
            return np.array([0.4, 0.0, 0.3]), np.array([1.0, 0.0, 0.0])
        
        print("? Resetting robot to initial position...")
        #给jaka构建命令去homepoint
        # self._robot.move_linear_extend(self._robot_initpos,_speed=self._robot_speed,_accel=self._robot_acc)
        
        new_robot_cartesian=self._robot.current_cartesian
        new_robot_joints=self._robot.current_joints
        # print(f"    new_robot_cartesian:{new_robot_cartesian}")
        # print(f"    new_robot_pos:{new_robot_cartesian[:3]}")
        # print(f"    new_robot_euler:{new_robot_cartesian[3:]}")
       
        
        robot_pos=0.001*np.array(new_robot_cartesian[:3])
        robot_euler=np.array(new_robot_cartesian[3:])
        robot_joint=np.array(new_robot_joints)
        print(f"? Robot reset complete")
        print(f"   Position: [{robot_pos[0]:.6f}, {robot_pos[1]:.6f}, {robot_pos[2]:.6f}]")
        print(f"  euler: [{robot_euler[0]:.6f}, {robot_euler[1]:.6f}, {robot_euler[2]:.6f}]")
        
     
        
        #返回 末端位姿,四元数,关节角,总之就是返回机器人当前位姿
       
        return robot_pos,robot_euler,robot_joint
    
    def  velocity_to_position_target(self, velocity_action):
        """Convert velocity action to position target for deoxys control
        
        This implements DROID-style velocity-to-delta conversion:
        - Velocities are normalized to [-1, 1] range
        - Scaled by max_delta parameters (not just dt)
        - Matches DROID's IK solver behavior
        """
        # Extract components
        lin_vel = velocity_action[:3]
        rot_vel = velocity_action[3:6]
        gripper_vel = velocity_action[6]
        
        # DROID-style velocity to delta conversion
        # Velocities are already clipped to [-1, 1] and represent fraction of max velocity
        # Scale by max_delta (which incorporates the control frequency)
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        
        # Apply DROID's scaling
        if lin_vel_norm > 1:
            lin_vel = lin_vel / lin_vel_norm
        if rot_vel_norm > 1:
            rot_vel = rot_vel / rot_vel_norm
            
        # Convert to position delta using DROID's max_delta parameters
        pos_delta = lin_vel * self.max_lin_delta
        rot_delta = rot_vel * self.max_rot_delta
        # Gripper uses simple scaling
        gripper_delta = gripper_vel * self.control_interval
        
        # Calculate target gripper
        target_gripper = np.clip(self.robot_gripper + gripper_delta, 0.0, 1.0)
        
      
               
            
     
        
        # rot_temp = quat_to_euler(rot_delta)
        # rot_delta=rot_temp[:3]
        # return target_pos, target_quat, target_gripper
        return pos_delta, rot_delta, target_gripper
    
    def control_loop(self):
        """Main control loop - now coordinates async workers"""
        message_count = 0
        last_debug_time = time.time()
        
        # Initialize robot on first frame
        if self.is_first_frame:
            # init_pos, init_euler,init_joint_positions = self.reset_robot()
            # print(f" init_euler: {init_euler}")
            # self.robot_pos = init_pos
            # self.robot_quat = euler_to_quat(init_euler)
            # self.robot_euler = init_euler
            # self.robot_gripper = 0.0
            # self.robot_joint_positions = init_joint_positions
            self.is_first_frame = False
            
            left_init_states=self.left_arm.update_state()
            right_init_states=self.right_arm.update_state()
            self.arm_states={
                "left":left_init_states,
                "right":right_init_states
            }
            # Store initial robot state
            with self._robot_state_lock:
                self._latest_robot_state = {
                    'left':self.arm_states["left"].copy(),
                    'right':self.arm_states["right"].copy(),
                }
        # Start camera manager if enabled
        if self.camera_manager:
            try:
                self.camera_manager.start()
                print("? Camera manager started")
            except Exception as e:
                print(f"??  Failed to start camera manager: {e}")
                # Continue without cameras
                self.camera_manager = None
        
        # Start worker threads
        if self.enable_recording and self.data_recorder:
            self._mcap_writer_thread = threading.Thread(target=self._mcap_writer_worker)
            self._mcap_writer_thread.daemon = True
            self._mcap_writer_thread.start()
            self._threads.append(self._mcap_writer_thread)
            print("? MCAP writer thread started")
            
            # Start data recording thread
            self._data_recording_thread = threading.Thread(target=self._data_recording_worker)
            self._data_recording_thread.daemon = True
            self._data_recording_thread.start()
            self._threads.append(self._data_recording_thread)
            print("? Data recording thread started")
     
        
        self._robot_control_thread = threading.Thread(target=self._robot_control_worker)
        self._robot_control_thread.daemon = True
        self._robot_control_thread.start()
        self._threads.append(self._robot_control_thread)
        print("? Robot control thread started")
        
        # Main loop now status and handles high-level control
        while self.running:
            try:
                current_time = time.time()
                
                # Handle robot reset after calibration
                if hasattr(self, '_reset_robot_after_calibration') and self._reset_robot_after_calibration:
                    self._reset_robot_after_calibration = False
                    print("? Executing robot reset after calibration...")
                    
                    # Pause control thread during reset
                    self._control_paused = True
                    time.sleep(0.1)  # Give control thread time to pause
                    
                    left_reset_states=self.left_arm.update_state()
                    right_reset_states=self.right_arm.update_state()
                    self.arm_states={
                        "left":left_reset_states,
                        "right":right_reset_states
                    }
                    # Store initial robot state
                    with self._robot_state_lock:
                        self._latest_robot_state = {
                            'left':self.arm_states["left"].copy(),
                            'right':self.arm_states["right"].copy(),
                        }
                    # self.robot_joint_positions = reset_joint_positions
                    
                    # Clear any pending actions
                    self.reset_origin = True
                    
                    # Resume control thread
                    self._control_paused = False
                    
                    print("? Robot is now at home position, ready for teleoperation")
                    print("   Hold grip button to start controlling the robot\n")
                
                # Debug output
                if self.debug and current_time - last_debug_time > 1.0:
                    with self._vr_state_lock:
                        vr_state = self._latest_vr_state
                    with self._robot_state_lock:
                        robot_state = self._latest_robot_state
                    
                    if vr_state and robot_state:
                        print(f"\n? Async Status [{message_count:04d}]:")
                        print(f"   VR State: {vr_state.timestamp:.3f}s ago")
                        print(f"   Robot State: {robot_state.timestamp:.3f}s ago")
                        print(f"   MCAP Queue: {self.mcap_queue.qsize()} items")
                        # print(f"   Recording: {'ACTIVE' if self.recording_active else 'INACTIVE'}")
                        
                    last_debug_time = current_time
                    message_count += 1
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.01)
                
            except Exception as e:
                if self.running:
                    print(f"? Error in main loop: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)
    
    def start(self):
        """Start the server"""
        try:
            # Run control loop
            self.control_loop()
        except KeyboardInterrupt:
            print("\n? Keyboard interrupt received")
            self.stop_server()
    
    def stop_server(self):
        """Gracefully stop the server"""
       
        if not self.running:
            return
            
        print("? Stopping Oculus VR Server...")
        self.running = False
        
        # Stop any active recording
        if self.recording_active and self.data_recorder:
            print("? Stopping active recording...")
            self.data_recorder.stop_recording(success=False)
            self.recording_active = False
        # Send poison pill to MCAP writer
        if self._mcap_writer_thread and self._mcap_writer_thread.is_alive():
            self.mcap_queue.put(None)
        
        # # Send poison pill to robot comm thread
        # if self._robot_comm_thread and self._robot_comm_thread.is_alive():
        #     self._robot_command_queue.put(None)
        
        # Stop state thread
        if hasattr(self, '_state_thread'):
            self._state_thread.join(timeout=1.0)
        
        # Stop worker threads
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        # Stop Oculus Reader
        if hasattr(self, 'oculus_reader'):
            try:
                self.oculus_reader.stop()
                print("? Oculus Reader stopped")
            except Exception as e:
                print(f"??  Error stopping Oculus Reader: {e}")
        
        # Stop camera manager
        if self.camera_manager:
            try:
                self.camera_manager.stop()
                print("? Camera manager stopped")
            except Exception as e:
                print(f"??  Error stopping camera manager: {e}")
        
        # Stop simulation server if running
        if self.sim_server:
            self.sim_server.stop()
            print("? Simulation server stopped")
        
        # Close robot connections
        if not self.debug:
            if hasattr(self, 'action_socket'):
                self.action_socket.close()
            if hasattr(self, 'controller_publisher'):
                self.controller_publisher.close()
            if hasattr(self, 'context'):
                self.context.term()
            print("? Robot connections and ZMQ resources closed")
        
        print("? Server stopped gracefully")
        sys.exit(0)

    def _mcap_writer_worker(self):
        """Asynchronous MCAP writer thread - processes data without blocking control"""
        print("? MCAP writer thread started")
        
        while self.running or not self.mcap_queue.empty():
            try:
                # Get data from queue with timeout
                timestep_data = self.mcap_queue.get(timeout=0.1)
                
                if timestep_data is None:  # Poison pill
                    break
                
                # Create timestep in Labelbox Robotics format
                timestep = {
                    "observation": {
                        "timestamp": {
                            "robot_state": {
                                "read_start": int(timestep_data.timestamp * 1e9),
                                "read_end": int(timestep_data.timestamp * 1e9)
                            }
                        },
                        "robot_state": {
                            "joint_positions": timestep_data.robot_state.joint_positions.tolist() if timestep_data.robot_state.joint_positions is not None else [],
                            "joint_velocities": [],
                            "joint_efforts": [],
                            "cartesian_position": np.concatenate([
                                timestep_data.robot_state.pos,
                                timestep_data.robot_state.euler
                            ]).tolist(),
                            "cartesian_velocity": [],
                            "gripper_position": timestep_data.robot_state.gripper,
                            "gripper_velocity": 0.0
                        },
                        "controller_info": timestep_data.info
                    },
                    "action": timestep_data.action.tolist() if hasattr(timestep_data.action, 'tolist') else timestep_data.action
                }
                
                # Write to MCAP
                self.data_recorder.write_timestep(timestep, timestep_data.timestamp)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"? Error in MCAP writer: {e}")
                import traceback
                traceback.print_exc()
        
        print("? MCAP writer thread stopped")

    def _robot_control_worker(self):
        """Asynchronous robot control thread - sends commands at control frequency"""
        print("? Robot control thread started")
        print(f"   Target control frequency: {self.control_hz}Hz")
        print(f"   Control interval: {self.control_interval*1000:.1f}ms")
        
        last_control_time = time.time()
        control_count = 0
        freq_check_time = time.time()
        
        # Track prediction accuracy
        prediction_errors = deque(maxlen=100)
        using_predictions = False
        
        while self.running:
            try:
                current_time = time.time()
                #线程之间串行并发导致控制帧率改变
                # Skip if control is paused
                if self._control_paused:
                    time.sleep(0.01)
                    continue
                
                # Control at specified frequency
                if current_time - last_control_time >= self.control_interval:#确保最小执行间隔
                    #获取最新 状态
                    # Get latest VR state
                    with self._vr_state_lock:
                        vr_state = self._latest_vr_state.copy() if self._latest_vr_state else None
                    
                    # Get latest robot state
                    with self._robot_state_lock:
                        robot_state = self._latest_robot_state.copy() if self._latest_robot_state else None
                    
                    if vr_state and robot_state:
                        # Check if we're using predictions
                        state_age = max(current_time - robot_state['left'].timestamp,
                                        current_time - robot_state['right'].timestamp)
                        if state_age > self.control_interval * 2:
                            #当前时间与采样时间差大于2倍控制间隔，说明机器人状态更新不及时，在debug信息中显示但并无实际调整
                            if not using_predictions:
                                using_predictions = True
                        else:
                            if using_predictions:
                                using_predictions = False
                        # 发给控制器
                        # Process control command
                        self._process_control_cycle(vr_state, robot_state, current_time)
                        control_count += 1
                    
                    last_control_time = current_time
                    
                    # Print actual frequency every second
                    
                    if current_time - freq_check_time >= 1.0:#每秒算一次
                        actual_freq = control_count / (current_time - freq_check_time)
                        if self.recording_active:
                            status = "PREDICTIVE" if using_predictions else "REAL-TIME"
                            print(f"? Control frequency: {actual_freq:.1f}Hz (target: {self.control_hz}Hz) - {status}")
                        control_count = 0
                        freq_check_time = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
            except Exception as e:
                if self.running:
                    print(f"? Error in robot control: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
        
        print("? Robot control thread stopped")

    def _data_recording_worker(self):
        """Records data at target frequency independent of robot control"""
        print("? Data recording thread started")
        print(f"   Target recording frequency: {self.recording_hz}Hz")
        
        last_record_time = time.time()
        record_count = 0
        freq_check_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Record at specified frequency
                if current_time - last_record_time >= self.recording_interval:
                    # Only record if recording is active
                    if self.recording_active and self.data_recorder:
                        # Get latest states
                        with self._vr_state_lock:
                            vr_state = self._latest_vr_state.copy() if self._latest_vr_state else None
                        
                        with self._robot_state_lock:
                            robot_state = self._latest_robot_state.copy() if self._latest_robot_state else None
                        
                        if vr_state and robot_state:
                            # Get current action (may be zero if not moving)
                            info = {
                                "success": vr_state.buttons.get("A", False) if self.controller_id == 'r' else vr_state.buttons.get("X", False),
                                "failure": vr_state.buttons.get("B", False) if self.controller_id == 'r' else vr_state.buttons.get("Y", False),
                                "movement_enabled": vr_state.movement_enabled,
                                "controller_on": vr_state.controller_on,
                                "poses": vr_state.poses,
                                "buttons": vr_state.buttons
                            }
                            
                            # Calculate action if movement is enabled
                            action = np.zeros(7)  # Default no movement
                            if vr_state.movement_enabled and hasattr(self, 'vr_state') and self.vr_state:
                                # Use the last calculated action if available
                                if hasattr(self, '_last_action'):
                                    action = self._last_action
                            
                            # Create timestep data
                            timestep_data = TimestepData(
                                timestamp=current_time,
                                vr_state=vr_state,
                                robot_state=robot_state,
                                action=action.copy(),
                                info=copy.deepcopy(info)
                            )
                            
                            # Queue for MCAP writer
                            try:
                                self.mcap_queue.put_nowait(timestep_data)
                                record_count += 1
                            except queue.Full:
                                print("??  MCAP queue full, dropping frame")
                    
                    last_record_time = current_time
                    
                    # Print actual frequency every second
                    if current_time - freq_check_time >= 1.0:
                        if self.recording_active:
                            actual_freq = record_count / (current_time - freq_check_time)
                            print(f"? Recording frequency: {actual_freq:.1f}Hz (target: {self.recording_hz}Hz)")
                        record_count = 0
                        freq_check_time = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
            except Exception as e:
                if self.running:
                    print(f"? Error in data recording: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
        
        print("? Data recording thread stopped")

    def _robot_comm_worker(self):
        """Handles robot communication asynchronously to prevent blocking control thread"""
        print("? Robot communication thread started")
        
        comm_count = 0
        total_comm_time = 0
        
        while self.running:
            try:
                # Get command from queue with timeout
                command = self._robot_command_queue.get(timeout=0.01)
                
                if command is None:  # Poison pill
                    break
                
                # Send command and receive response
                comm_start = time.time()
                with self._robot_comm_lock:
                    self.action_socket.send(bytes(pickle.dumps(command, protocol=-1)))
                    response = pickle.loads(self.action_socket.recv())
                comm_time = time.time() - comm_start
                
                comm_count += 1
                total_comm_time += comm_time
                
                # Log communication stats periodically
                if comm_count % 10 == 0:
                    avg_comm_time = total_comm_time / comm_count
                    print(f"? Avg robot comm: {avg_comm_time*1000:.1f}ms")
                
                # Put response in queue
                try:
                    self._robot_response_queue.put_nowait(response)
                except queue.Full:
                    # Drop oldest response if queue is full
                    try:
                        self._robot_response_queue.get_nowait()
                        self._robot_response_queue.put_nowait(response)
                    except:
                        pass
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.running:
                    print(f"? Error in robot communication: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
        
        print("? Robot communication thread stopped")

    def _process_control_cycle(self, vr_state: VRState, robot_state, current_time: float):
        """Process a single control cycle with given VR and robot states"""
        # Restore state from thread-safe structures
        #self._state？
        #获取最新的机器人vr状态
        
        self._state["poses"] = vr_state.poses
        self._state["buttons"] = vr_state.buttons
        #这里的的poses 和 buttons 是包含左右手柄数据的字典
        #
        self._state["movement_enabled"] = vr_state.movement_enabled
        self._state["controller_on"] = vr_state.controller_on
      
        # self._state["grip_released"] = vr_state.grip_released
        # self._state["toggle_state"] = vr_state.toggle_state
        # print(f"    grip_released state: {self._state['grip_released']}")
        # Update robot state
        #这里的状态要两份
        # self.robot_pos = robot_state.pos
        # self.robot_quat = robot_state.quat
        # self.robot_euler = robot_state.euler
        # self.robot_gripper = robot_state.gripper
        # self.robot_joint_positions = robot_state.joint_positions
        
        
        self.robot_pos ={'left': robot_state['left'].pos,
                         'right': robot_state['right'].pos } 
        self.robot_quat = {'left': robot_state['left'].quat,
                         'right': robot_state['right'].quat } 
        self.robot_euler = {'left': robot_state['left'].euler,
                         'right': robot_state['right'].euler } 
        self.robot_gripper =  {'left': robot_state['left'].gripper,
                         'right': robot_state['right'].gripper  } 
        self.robot_joint_positions =  {'left': robot_state['left'].joint_positions,
                                     'right': robot_state['right'].joint_positions  } 
        
        
        
        # Get controller info
        #主控制器信息 用于全局控制录制数据，但movementable用于单臂
        info = self.get_info()
        
        #录制按键处理
        if self.enable_recording and self.data_recorder:
            # A button: Start/Reset recording
            current_a_button = info["success"]
            if current_a_button and not self.prev_a_button:  # Rising edge
                if self.recording_active:
                    # Stop current recording
                    print("\n? A button pressed - Stopping current recording...")
                    self.data_recorder.reset_recording()
                    self.recording_active = False
                    print("? Recording stopped (not saved)")
                    print("   Press A to start a new recording")
                else:
                    # Start recording
                    print("\n??  A button pressed - Starting recording...")
                    self.data_recorder.start_recording()
                    self.recording_active = True
                    print("? Recording started")
                    print("   Press A again to stop/discard")
                    print("   Press B to mark as successful and save")
            self.prev_a_button = current_a_button
            
            # B button: Mark as successful and stop
            if info["failure"] and self.recording_active:
                print("\n? B button pressed - Marking recording as successful...")
                saved_filepath = self.data_recorder.stop_recording(success=True)
                self.recording_active = False
                print("? Recording saved successfully")
                
                # Verify data if requested
                if self.verify_data and saved_filepath:
                    print("\n? Verifying recorded data...")
                    try:
                        verifier = MCAPVerifier(saved_filepath)
                        results = verifier.verify(verbose=True)
                        
                        # Check if verification passed
                        if not results["summary"]["is_valid"]:
                            print("\n??  WARNING: Data verification found issues!")
                            print("   The recording may not be suitable for training.")
                    except Exception as e:
                        print(f"\n? Error during verification: {e}")
                        import traceback
                        traceback.print_exc()
                
                print("\n   Press A to start a new recording")
        else:
            # Original behavior when recording is disabled
            if info["success"]:
                print("\n? Success button pressed!")
                if not self.debug:
                    # Send termination signal
                    self.stop_server()
                    return
            
            if info["failure"]:
                print("\n? Failure button pressed!")
                if not self.debug:
                    # Send termination signal
                    self.stop_server()
                    return
        
        # Default action (no movement)
        action = np.zeros(14)
        action_info = {}
        # if  self._state["toggle_state"]:
        #     if self._robot_gripper_count==0:
        #         print("    Gripper opened")
        #         self._robot.set_gripper_params(position=1000, force=30, speed=50,block=0)
        #     else:
        #         print("    Gripper closed")
        #         self._robot.grasp(timeout=8)
        #     self._state["toggle_state"]=False

        # #独立控制rz
        # joyvalue=self._state["buttons"].get("rightJS" if self.right_controller else "leftJS", [0.0])[0]
        # # print("    joyvalue:",joyvalue)
        # if abs(joyvalue)>0.5:    
        #     if joyvalue>0:          rz_delta=np.asarray([0,0,0,0,0,radians(0.5)])
        #     else:                   rz_delta=np.asarray([0,0,0,0,0,-radians(0.5)])
        #     self._robot.move_linear_extend(rz_delta, _speed=200, _accel=100,TOL=0.1,_is_block=False,_move_mode=1)
        # # Calculate action if movement is enabled
         # Read Sensor
        if self.update_sensor:
            self._process_reading()#处理两个手柄的信息
            self.update_sensor = False
        
        # Check if we have valid data
        if self.vr_state is None or self.robot_pos is None:
            return np.zeros(14), {}
        
        if  info["left_movement_enabled"] and self._state["poses"]:
            #vr位姿更新并且运动使能
            self.left_arm.robot.servo_move_enable(True)
            
            if self.reset_origin_left:
                self.reset_origin_left = {"pos": self.robot_pos["left"], "quat": self.robot_quat["left"],"euler":self.robot_euler["left"]}
                self.reset_origin_left= {"pos": self.vr_state["pos"]["left"], "quat": self.vr_state["quat"]["left"],"euler":self.vr_state["euler"]["left"]}
                self.reset_origin_left = False
                print("? Origin calibrated")
           
                action_left= self.left_arm.calculate_action(
                        vr_pos=self.vr_state["pos"]["left"],
                        vr_euler=self.vr_state["euler"]["left"],
                        vr_quat=self.vr_state["quat"]["left"],
                        vr_gripper=self.vr_state["gripper"],
                        robot_pos=self.robot_pos["left"],
                        robot_quat=self.robot_quat["left"],
                        vr_origin_pos= self.robot_origin_left["pos"] , 
                        vr_origin_euler=self.robot_origin_left["euler"],
                    )
            
            
            # Convert velocity to position target
            target_pos, target_euler, target_gripper = self.velocity_to_position_target( velocity_action=action_left )
            pos_scale =1
            euler_scale = 1
           
            delta=np.asarray([target_pos[0]*1000*pos_scale,target_pos[1]*1000*pos_scale,target_pos[2]*1000*pos_scale,-target_euler[0]*euler_scale,-target_euler[1]*euler_scale,target_euler[2]*euler_scale])
           

            # Send action to robot (or simulate)
            if not self.debug:
                # Send action to robot - DEOXYS EXPECTS QUATERNIONS
                _step_num=5
                if self.enable_performance_mode:_step_num=2
                
                try:
                    # pass
                    self.left_arm.robot.servo_p_extend(delta, move_mode=1, step_num=_step_num)
                    
                except Exception as e:
                    print(f"    ? Error in robot move: {e}")        
        else:
            # new_left_robot_state = robot_state["left"]
            action_left = np.zeros(7)
            if not self.debug and self._state["left_grip_released"]:
                print("    left Gripper released  Stop!") 
                self.left_arm.robot.servo_move_enable(False)
                self.left_arm.stop_motion()
                self._state["left_grip_released"]=False

       
            
                          
        if  info["right_movement_enabled"] and self._state["poses"]:
            #vr位姿更新并且运动使能
            self.right_arm.robot.servo_move_enable(True)
            
            if self.reset_origin_right:
                self.robot_origin_right = {"pos": self.robot_pos["right"], "quat": self.robot_quat["right"],"euler":self.robot_euler["right"]}
                self.vr_origin_right= {"pos": self.vr_state["pos"]["right"], "quat": self.vr_state["quat"]["right"],"euler":self.vr_state["euler"]["right"]}
                self.reset_origin_right = False
                print("? Origin calibrated")
           
                action_right= self.right_arm.calculate_action(
                        vr_pos=self.vr_state["pos"]["right"],
                        vr_euler=self.vr_state["euler"]["right"],
                        vr_quat=self.vr_state["quat"]["right"],
                        vr_gripper=self.vr_state["gripper"],
                        robot_pos=self.robot_pos["right"],
                        robot_quat=self.robot_quat["right"],
                        vr_origin_pos= self.robot_origin_right["pos"] , 
                        vr_origin_euler=self.robot_origin_right["euler"],
                    )
          
            
            # Convert velocity to position target
            target_pos, target_euler, target_gripper = self.velocity_to_position_target( velocity_action=action_right )
            pos_scale =1
            euler_scale = 1
           
            delta=np.asarray([target_pos[0]*1000*pos_scale,target_pos[1]*1000*pos_scale,target_pos[2]*1000*pos_scale,-target_euler[0]*euler_scale,-target_euler[1]*euler_scale,target_euler[2]*euler_scale])
           

            # Send action to robot (or simulate)
            if not self.debug:
                # Send action to robot - DEOXYS EXPECTS QUATERNIONS
                _step_num=5
                if self.enable_performance_mode:_step_num=2
                
                try:
                    self.right_arm.robot.servo_p_extend(delta, move_mode=1, step_num=_step_num)   
                except Exception as e:
                    print(f"    ? Error in robot move: {e}")
                
                   
        else:
            # new_left_robot_state = robot_state["left"]
            action_right = np.zeros(7)
            if not self.debug and self._state["right_grip_released"]:
                print("    right Gripper released  Stop!") 
                self.right_arm.robot.servo_move_enable(False)
                self.right_arm.stop_motion()
                self._state["right_grip_released"]=False        
                
                
        self._last_action = np.concatenate([action_right.copy(), action_left.copy()])
            
        #状态更新  如果有一个臂使能  就更新状态 
        if  info["right_movement_enabled"] or info["left_movement_enabled"]:
            # Not moving - use current robot state
            try:
                    # 更新左右臂的状态
                new_left_robot_cartesian = np.array(self.left_arm.robot.current_cartesian)
                new_left_robot_joints = np.array(self.left_arm.robot.current_joints)
                new_right_robot_cartesian = np.array(self.right_arm.robot.current_cartesian)
                new_right_robot_joints = np.array(self.right_arm.robot.current_joints)

                # 创建新的左右臂状态
                new_left_robot_state = RobotState(
                    timestamp=current_time,
                    pos=0.001*new_left_robot_cartesian[:3],
                    quat=euler_to_quat(new_left_robot_cartesian[3:]),
                    euler=new_left_robot_cartesian[3:],
                    gripper=self.left_arm._robot_gripper_state,
                    joint_positions=new_left_robot_joints        
                )

                new_right_robot_state = RobotState(
                    timestamp=current_time,
                    pos=0.001*new_right_robot_cartesian[:3],
                    quat=euler_to_quat(new_right_robot_cartesian[3:]),
                    euler=new_right_robot_cartesian[3:],
                    gripper=self.right_arm._robot_gripper_state,
                    joint_positions=new_right_robot_joints        
                )

                # 使用线程锁更新机器人状态
                with self._robot_state_lock:
                    self._latest_robot_state = {
                        'left': new_left_robot_state,
                        'right': new_right_robot_state
                    }

                # 更新局部状态以供下次计算使用
                self.robot_pos = {'left': new_left_robot_state.pos, 'right': new_right_robot_state.pos}
                self.robot_quat = {'left': new_left_robot_state.quat, 'right': new_right_robot_state.quat}
                self.robot_euler = {'left': new_left_robot_state.euler, 'right': new_right_robot_state.euler}
                self.robot_gripper = {'left': new_left_robot_state.gripper, 'right': new_right_robot_state.gripper}
                self.robot_joint_positions = {'left': new_left_robot_state.joint_positions, 'right': new_right_robot_state.joint_positions}
            except Exception as e:
                print(f"    ? Error in robot_state update: {e}") 
            
            
        else:   
            new_robot_state = robot_state
            with self._robot_state_lock:    
                self._latest_robot_state = new_robot_state

      
          
        
        # Note: Data recording is now handled by the dedicated recording thread
        # which runs at the target frequency independent of robot control


def main():
    parser = argparse.ArgumentParser(
        description='Oculus VR Server with DROID-exact VRPolicy Control',
        epilog='''
This server implements the exact VRPolicy control from DROID with intuitive calibration.

Features:
  - DROID-exact control parameters (gains, velocities, transforms)
  - Intuitive forward direction calibration (hold joystick + move)
  - Deoxys-compatible quaternion handling
  - Origin recalibration on grip press/release
  - MCAP data recording in DROID-compatible format

Calibration:
  - Hold joystick button and move controller forward (at least 3mm)
  - The direction you move defines "forward" for the robot
  - Release joystick to complete calibration
  - Falls back to DROID-style calibration if no movement detected

Recording (when enabled):
  - Press A button to start recording or stop current recording
  - Press B button to mark recording as successful and save
  - Recordings are saved in ~/recordings/success
  - Stopped recordings (via A button) are discarded

Note: This version is adapted for Deoxys control (quaternion-based) instead of
Polymetis (euler angle-based). The rotation handling has been adjusted accordingly.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode (no robot control)')
    parser.add_argument('--left-controller', action='store_true',
                        help='Use left controller instead of right (default: right)')
    parser.add_argument('--ip', type=str, default=None,
                        help='IP address of Quest device (default: USB connection)')
    parser.add_argument('--simulation', action='store_true',
                        help='Use simulated FR3 robot instead of real hardware')
    parser.add_argument('--coord-transform', nargs='+', type=float,
                        help='Custom coordinate transformation vector (format: x y z w)')
    parser.add_argument('--rotation-mode', type=str, default='labelbox',
                        choices=['labelbox'],
                        help='Rotation mapping mode (default: labelbox)')
    parser.add_argument('--hot-reload', action='store_true',
                        help='Enable hot reload mode (auto-restart on file changes)')
    parser.add_argument('--performance', action='store_true',
                        help='Enable performance mode for tighter tracking (2x frequency, higher gains)')
    parser.add_argument('--no-recording', action='store_true',
                        help='Disable MCAP data recording functionality')
    parser.add_argument('--verify-data', action='store_true',
                        help='Verify MCAP data integrity after successful recording')
    parser.add_argument('--camera-config', type=str, default=None,
                        help='Path to camera configuration YAML file (e.g., configs/cameras.yaml)')
    parser.add_argument('--enable-cameras', action='store_true',
                        help='Enable camera recording with MCAP data')
    parser.add_argument('--auto-discover-cameras', action='store_true',
                        help='Automatically discover and use all connected cameras')
    
    args = parser.parse_args()
    
    # If hot reload is requested, launch the hot reload wrapper instead
    # if args.hot_reload:
    #     import subprocess
    #     import sys
        
    #     # Remove --hot-reload from args and pass the rest to the wrapper
    #     new_args = [arg for arg in sys.argv[1:] if arg != '--hot-reload']
        
    #     print("? Launching in hot reload mode...")
        
    #     # Check if hot reload script exists
    #     if not os.path.exists('oculus_vr_server_hotreload.py'):
    #         print("? Hot reload script not found!")
    #         print("   Make sure oculus_vr_server_hotreload.py is in the same directory")
    #         sys.exit(1)
        
    #     # Launch the hot reload wrapper
    #     try:
    #         subprocess.run([sys.executable, 'oculus_vr_server_hotreload.py'] + new_args)
    #     except KeyboardInterrupt:
    #         print("\n? Hot reload stopped")
    #     sys.exit(0)
    
    # Handle auto-discovery of cameras
    # if args.auto_discover_cameras:
    # if True:
    #     print("? Auto-discovering cameras...")
    #     try:
    #         from Modules.camera_utils import discover_all_cameras, generate_camera_config
            
    #         cameras = discover_all_cameras()
    #         if cameras:
    #             # Generate temporary config
    #             temp_config = "/tmp/cameras_autodiscovered.yaml"
    #             generate_camera_config(cameras, temp_config)
                
    #             # Override camera config path
    #             args.camera_config = temp_config
    #             args.enable_cameras = True
                
    #             print(f"? Using auto-discovered cameras from: {temp_config}")
    #         else:
    #             print("??  No cameras found during auto-discovery")
    #     except Exception as e:
    #         print(f"? Camera auto-discovery failed: {e}")
    
    # Load camera configuration if provided
    camera_configs = None
    if args.camera_config:
        try:
            import yaml
            with open(args.camera_config, 'r') as f:
                camera_configs = yaml.safe_load(f)
            print(f"? Loaded camera configuration from {args.camera_config}")
        except Exception as e:
            print(f"??  Failed to load camera config: {e}")
            print("   Continuing without camera configuration")
    
    # Normal execution (no hot reload)
    # Create and start server with DROID-exact parameters
    coord_transform = args.coord_transform
    # server = OculusVRServer(
    #     debug=args.debug,
    #     right_controller=not args.left_controller,
    #     ip_address=args.ip,
    #     simulation=args.simulation,
    #     coord_transform=coord_transform,
    #     rotation_mode=args.rotation_mode,
    #     performance_mode=args.performance,
    #     enable_recording=not args.no_recording,
    #     camera_configs=camera_configs,
    #     verify_data=args.verify_data,
    #     camera_config_path=args.camera_config,
    #     enable_cameras=args.enable_cameras
    # )
    server = OculusVRServer(
        debug=args.debug,
        right_controller=not args.left_controller,
        ip_address=args.ip,
        simulation=args.simulation,
        coord_transform=coord_transform,
        rotation_mode=args.rotation_mode,
        performance_mode=args.performance,
        enable_recording=not args.no_recording,
        
        verify_data=args.verify_data,
        camera_config_path='configs/cameras_intel.yaml',
        enable_cameras=True
    )
    try:
        server.start()
    except Exception as e:
        print(f"? Unexpected error: {e}")
        import traceback
        traceback.print_exc()   
        server.stop_server()
        server._robot.robot.servo_move_enable(False)


if __name__ == "__main__":
    main() 