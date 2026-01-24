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
    def __init__(self,side:str,ip_address:str,debug=False,performance=False):
        self.side=side    #这个臂是左臂还是右臂
        self.controller_key="l" if side=="left" else "r"
        self.ip=ip_address
        self.debug=debug
        self.performance=performance
        
        #state
        self.robot=None
        self.current_state:Optional[RobotState] =None   # 表明self.current_state是一个可选RobotState对象，在没有输入时为NONE
        
        
        
        
        
        
        self.init_robot()
        
    def init_robot(self):
       
        if self.debug: #调试模式
            print(f"{self.side} arm initializing ...")
            return
        print(f"[{self.side}] Connecting to {self.ip}...")
        try:
            self.robot = JakaRobot(self.ip)
            if self.robot.connect(0.02)[0]:
                self.robot.power_on()
                self.robot.enable()
                self.robot.set_tool_data(140)
                self.robot.init_gripper(0x01)
                self.robot.set_gripper_params(position=1000, force=30, speed=50, block=1)
                
                self.robot.robot.servo_move_enable(True)
                print(f"[{self.side}] ? Ready.")
            else:
                print(f"[{self.side}] ? Connection Failed.")
        except Exception as e:
            print(f"[{self.side}] ? Exception: {e}")
               
    def update_state(self):
        now=time.time()
        if self.debug:
            print (f"{self.side} arm updating state...")
            return
        try:
            #获取当前位姿
            cart=np.array(self.robot.current_cartesian)
            joints=np.array(self.robot.current_joints)
            
            #单位转换，Jaka返回的位姿是单位是mm，旋转是弧度，记录数据的时候要用米和弧度
            pos=0.001*cart[:3]
            euler=cart[3:]
            quat=euler_to_quat(euler)
            
            self.current_state=RobotState(
                                timestamp=now,
                                pos=pos,
                                quat=quat,
                                euler=euler,
                                gripper=self.robot.gripper_state,
                                joint_positions=joints)
            
            return self.current_state
            
        except Exception as e:
            print(f"[{self.side}] arm read error:{e}") 
    def step_control(self,vr_poses:Dict,vr_buttons:Dict):
        pass







    def send_servo_cmd(self,pos_action,rot_action):
        pass
    def transform_pose(self,raw_mat):