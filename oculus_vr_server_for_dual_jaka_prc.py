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

oculus VRserver 负责VR连接，机器人控制，数据录制，和位姿映射
重构后
oculusVRserver 只负责VR连接，数据录制，和位姿映射
新创建一个arm controller负责单臂的所有控制逻辑

'''
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

class ArmController:
    '''
    负责单臂的控制，状态更新
    
    '''
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