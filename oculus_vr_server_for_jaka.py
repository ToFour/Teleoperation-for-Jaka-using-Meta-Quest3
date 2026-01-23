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
                 enable_cameras=False):
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
        # self._robot=None
        self._robot_speed = 10
        self._robot_acc = 10
        self._robot_initpos = SAFE_POSITION
        self._robot_lastgripperState=GRIPPER_OPEN
        self._robot_gripper_count = 0
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
        if coord_transform is None:
            # Try a more standard transformation that should work for most robots
            # This maps: VR +Z (forward) → Robot +X (forward)
            #           VR +X (right) → Robot -Y (right) 
            #           VR +Y (up) → Robot +Z (up)
            #VR [z x y] robot [x y z  ]
            # rmat_reorder = [-3, -1, 2, 4]  # More intuitive default  当没进行校准时，这里能进行初步的坐标轴映射
            rmat_reorder = [2, 1, -3, 4] 
            print("\n⚠️  Using adjusted coordinate transformation for better compatibility")
            print("   If rotation is still incorrect, try --coord-transform with different values")
        else:
            rmat_reorder = coord_transform
            
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        
        # Rotation transformation matrix (separate from position)
        self.rotation_mode = rotation_mode
        # For labelbox mode, we apply euler angle transformations directly in _process_reading
        # to avoid creating invalid rotation matrices
        
        if self.debug or coord_transform is not None:
            print("\n🔍 Coordinate Transformation:")
            print(f"   Position reorder vector: {rmat_reorder}")
            print(f"   Rotation mode: {rotation_mode}")
            print("   Position Transformation Matrix:")
            for i in range(4):
                row = self.global_to_env_mat[i]
                print(f"   [{row[0]:6.1f}, {row[1]:6.1f}, {row[2]:6.1f}, {row[3]:6.1f}]")
            
            # Show what this transformation does
            test_vecs = [
                ([1, 0, 0, 1], "VR right (+X)"),
                ([0, 1, 0, 1], "VR up (+Y)"),
                ([0, 0, 1, 1], "VR forward (+Z)"),
            ]
            print("\n   Position Mapping:")
            for vec, name in test_vecs:
                transformed = self.global_to_env_mat @ np.array(vec)
                direction = ""
                if abs(transformed[0]) > 0.5:
                    direction = f"Robot {'forward' if transformed[0] > 0 else 'backward'} (X)"
                elif abs(transformed[1]) > 0.5:
                    direction = f"Robot {'left' if transformed[1] > 0 else 'right'} (Y)"
                elif abs(transformed[2]) > 0.5:
                    direction = f"Robot {'up' if transformed[2] > 0 else 'down'} (Z)"
                print(f"   {name} → {direction}")
            
            # Show rotation mapping
            print("\n   Rotation Mapping (Labelbox mode):")
            print("   VR Roll → Robot Roll (INVERTED for ergonomics)")
            print("   VR Pitch → Robot Pitch")
            print("   VR Yaw → Robot Yaw")
            print("   Axis transform: [X, Y, Z] → [-Y, X, Z]")
        
        # Initialize transformation matrices
        self.vr_to_global_mat = np.eye(4)
        
        # Controller ID
        self.controller_id = "r" if right_controller else "l"
        
        # Initialize state
        self.reset_state()
        
        # Initialize Oculus Reader
        print("🎮 Initializing Oculus Reader...")
        try:
            self.oculus_reader = OculusReader(
                ip_address=ip_address,
                print_FPS=False
            )
            print("✅ Oculus Reader initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Oculus Reader: {e}")
            sys.exit(1)

        if not self.debug:
            print("🤖 Connecting to robot...")
            try:
                success, message = self._robot.connect(0.02)
                if success:
                    #改为jaka上电使能
                    self._robot.power_on()
                    self._robot.enable()
                    self._robot.init_gripper(0x01)
                    self._robot.set_gripper_params(position=1000, force=30, speed=50,block=1)
                    self._robot.robot.set_tool_id(1)
                    self._robot.set_tool_data(140) 
                    self._robot.robot.servo_move_enable(True)
                    target_pos= [355, -122, 356, radians(180.0), radians(0.0), radians(145.0)]
                    self._robot.move_linear_extend(target_pos,_speed=200,_accel=100,TOL= 0.1,_is_block=True,_move_mode=0)
                    # self._robot.start_heartbeat(0.02)
                    print("✅ Connected to robot ")
                
              
             
            except Exception as e:
                print(f"❌ Failed to connect to robot: {e}")
                sys.exit(1)
       
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
                print("\n🔍 Testing camera functionality...")
                from Modules.camera_test import test_cameras
              
                # Load camera config
                import yaml
                with open(self.camera_config_path, 'r') as f:
                    test_camera_configs = yaml.safe_load(f)
                
                # Run camera tests
                all_passed, test_results = test_cameras(test_camera_configs)
                
                if not all_passed:
                    print("\n❌ Camera tests failed!")
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
                print("📷 Camera manager initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize camera manager: {e}")
                self.camera_manager = None
        
        if self.enable_recording:
            self.data_recorder = MCAPDataRecorder(
                camera_configs=camera_configs,
                save_images=True,
                save_depth=True,  # Enable depth recording if cameras support it
                camera_manager=self.camera_manager
            )
            print("📹 MCAP data recording enabled")
       
       
       
       
        # Setup signal handlers for graceful shutdown？
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Start state listening thread
        self._state_thread = threading.Thread(target=self._update_internal_state)
        self._state_thread.daemon = True
        self._state_thread.start()
        
        print("\n🎮 Oculus VR Server with DROID-exact VRPolicy Control")
        print(f"   Using {'RIGHT' if right_controller else 'LEFT'} controller")
        print(f"   Mode: {'DEBUG' if debug else 'LIVE ROBOT CONTROL'}")
        print(f"   Robot: {'SIMULATED FR3' if simulation else 'REAL HARDWARE'}")
        print(f"   Control frequency: {self.control_hz}Hz")
        print(f"   Position gain: {self.pos_action_gain}")
        print(f"   Rotation gain: {self.rot_action_gain}")
        print(f"   Gripper gain: {self.gripper_action_gain}")
        
        print("\n📋 Controls:")
        print("   - HOLD grip button: Enable teleoperation")
        print("   - RELEASE grip button: Pause teleoperation")
        print("   - PRESS trigger: Close gripper")
        print("   - RELEASE trigger: Open gripper")
        if self.enable_recording:
            print("\n📹 Recording Controls:")
            print("   - A button: Start recording or stop current recording")
            print("   - B button: Mark recording as successful and save")
            print("   - Recordings saved to: ~/recordings/success")
            print("   - Stopped recordings (via A button) are discarded")
        else:
            print("   - A/X button: Mark success and exit")
            print("   - B/Y button: Mark failure and exit")
        print("\n🧭 Forward Direction Calibration:")
        print("   - HOLD joystick button and MOVE controller forward")
        print("   - The direction you move defines 'forward' for the robot")
        print("   - Move at least 3mm in your desired forward direction")
        print("   - Release joystick button to complete calibration")
        
        print("\n💡 Tips:")
        print("   - Calibrate forward direction before starting teleoperation")
        print("   - Each grip press/release recalibrates the origin (starting position)")
        print("   - Robot movements are relative to the position when grip was pressed")
        
    
        
        print("\n⚠️  Note: Using Deoxys control interface (quaternion-based)")
        print("   Rotations are handled differently than DROID's Polymetis interface")
        
        print("\n💡 Hot Reload:")
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
            self.pos_action_gain = 15.0  # Even higher for tighter translation
            self.rot_action_gain = 1.5# Increased from 2.0
            
            # Adjust max deltas for higher frequency
            # Since we're running at 2x frequency, we can use smaller deltas per cycle
            # but achieve the same or better overall speed
            #调整每个cycle 的delta大小
            
            self.max_lin_delta = 0.003  # Maximum linear movement per control cycle
            self.max_rot_delta =0.008 
            print("\n⚡ PERFORMANCE MODE ENABLED:")
            print(f"   Control frequency: {self.control_hz}Hz (2x faster)")
            print(f"   Position gain: {self.pos_action_gain} (100% higher)")
            print(f"   Rotation gain: {self.rot_action_gain} (50% higher)")
            print(f"   Tighter tracking for better translation following")
        self.output_deadzone = 0.003
        self.output_euler_deadzone = radians(5)
        # Translation tracking improvements
        self.translation_deadzone = 0.005  # 0.5mm deadzone to filter noise
        self.use_position_filter = True  # Filter out noise in position tracking
        self.position_filter_alpha = 0.9  # Higher = more responsive, lower = more smooth
        self._last_vr_pos = None  # For position filtering
        
        # Additional DROID IK solver parameters for enhanced control
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
        self._state = {
            "poses": {},
            "buttons": {"A": False, "B": False, "X": False, "Y": False},
            "movement_enabled": False,
            "controller_on": True,
            "grip_released": False,
            "toggle_state": False
        }
        self.update_sensor = True
        self.reset_origin = True
        self.reset_orientation = True
        self.robot_origin = None
        self.vr_origin = None
        self.vr_state = None
        
        # Robot state - Deoxys uses quaternions directly
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
        
        # First frame flag
        self.is_first_frame = True
        
        # Flag to reset robot after calibration
        self._reset_robot_after_calibration = False
        
        # Added for rotation tracking
        self._last_controller_rot = None
        
        # Position filtering state
        self._last_vr_pos = None
        
        # Last action for recording thread
        self._last_action = np.zeros(7)
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C and other termination signals"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.stop_server()
        
    def _update_internal_state(self, num_wait_sec=5, hz=50):
        """Continuously poll VR controller state at 50Hz - with intuitive calibration"""
        last_read_time = time.time()
        
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
            current_grip = buttons.get(self.controller_id.upper() + "G", False)
            current_joystick = buttons.get(self.controller_id.upper() + "J", False)
            
            # Detect edge transitions
            grip_toggled = self.prev_grip_state != current_grip
            joystick_pressed = current_joystick and not self.prev_joystick_state  # Rising edge
            joystick_released = not current_joystick and self.prev_joystick_state  # Falling edge
            Grip_released = not current_grip and self.prev_grip_state  # Falling edge

            if Grip_released and self.debug==False:
                print("    Gripper released")
                self._state["grip_released"] = True
              
                # self._robot.robot.servo_move_enable(False)
                # self._robot.stop_motion()
                # self._robot.robot.servo_move_enable(False)
            trigger_value = buttons.get("rightTrig" if self.right_controller else "leftTrig", [0.0])[0]
            gripper_state = GRIPPER_CLOSE if trigger_value > 0.1 else GRIPPER_OPEN
            # print(f"    Gripper state: {'CLOSED' if gripper_state == GRIPPER_CLOSE else 'OPEN'}")
            toggle_state=gripper_state==GRIPPER_CLOSE and self._robot_lastgripperState ==GRIPPER_OPEN #上升沿
            if toggle_state:
                self._state["toggle_state"] = True
                self._robot_gripper_count+=1        
                if self._robot_gripper_count>1: self._robot_gripper_count=0
                print(f"    Gripper count: {self._robot_gripper_count}")
               
            self._robot_lastgripperState=gripper_state
            # Update control flags
            self.update_sensor = self.update_sensor or current_grip
            self.reset_origin = self.reset_origin or grip_toggled
            
            # Save Info
            self._state["poses"] = poses
            self._state["buttons"] = buttons
            self._state["movement_enabled"] = current_grip
            self._state["controller_on"] = True
            # self._state["grip_released"] = Grip_released
            # self._state["toggle_state"] = toggle_state
            last_read_time = time.time()
            
            # Publish VR state to async system
            current_time = time.time()
            vr_state = VRState(
                timestamp=current_time,
                poses=copy.deepcopy(poses),
                buttons=copy.deepcopy(buttons),
                movement_enabled=current_grip,
                controller_on=True,
                grip_released=Grip_released,
                toggle_state=toggle_state
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
                    print(f"\n🎯 Forward calibration started - Move controller in desired forward direction")
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
                            
                            print(f"\n✅ Forward direction calibrated!")
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
                            print("📐 Stored neutral controller orientation")
                            
                            # Reset robot to home position after calibration
                            if not self.debug and not self.reset_orientation:
                                print("🏠 Moving robot to reset position after calibration...")
                                self._reset_robot_after_calibration = True
                            elif self.debug:
                                print("\n✅ Calibration complete! Ready for teleoperation.")
                                print("   Hold grip button to start controlling the robot")
                        else:
                            print(f"\n⚠️  Not enough movement detected ({movement_distance*1000:.1f}mm)")
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
                stop_updating = self._state["buttons"][self.controller_id.upper() + "J"] or self._state["movement_enabled"]
                if stop_updating:
                    rot_mat = np.asarray(self._state["poses"][self.controller_id])
                    self.reset_orientation = False
                    # try to invert the rotation matrix, if not possible, then just use the identity matrix
                    try:
                        rot_mat = np.linalg.inv(rot_mat)
                    except:
                        print(f"exception for rot mat: {rot_mat}")
                        rot_mat = np.eye(4)
                        self.reset_orientation = True
                    self.vr_to_global_mat = rot_mat
                    print("📐 Orientation reset (DROID-style)")
                    
                    # Store the current controller pose as the neutral reference
                    # Store the raw pose, not transformed
                    self.vr_neutral_pose = np.asarray(self._state["poses"][self.controller_id]).copy()
                    print("📐 Stored neutral controller orientation")
                    
                    # Reset robot to home position after calibration
                    if not self.debug and not self.reset_orientation:
                        print("🏠 Moving robot to reset position after calibration...")
                        self._reset_robot_after_calibration = True
            
            # Update previous button states for next iteration
            self.prev_grip_state = current_grip
            self.prev_joystick_state = current_joystick
    
    def _process_reading(self):
        """Apply coordinate transformations to VR controller pose - DROID exact"""
        rot_mat = np.asarray(self._state["poses"][self.controller_id])
        
        # Apply position transformation
        transformed_mat = self.global_to_env_mat @ self.vr_to_global_mat @ rot_mat
        vr_pos = self.spatial_coeff * transformed_mat[:3, 3]
        
        # Apply position filtering to reduce noise/drift
        if self.use_position_filter and self._last_vr_pos is not None:
            # Calculate position change
            #对增量进行死区滤波
            pos_delta = vr_pos - self._last_vr_pos
            
            # Apply deadzone to filter out small movements (noise)
            for i in range(3):
                if abs(pos_delta[i]) < self.translation_deadzone:
                    pos_delta[i] = 0.0
            
            # Update position with filtered delta
            vr_pos = self._last_vr_pos + pos_delta
            
            # Alternative: Exponential moving average filter
            # vr_pos = self.position_filter_alpha * vr_pos + (1 - self.position_filter_alpha) * self._last_vr_pos
        
        self._last_vr_pos = vr_pos.copy()
        
        # Handle rotation
        if hasattr(self, 'vr_neutral_pose') and self.vr_neutral_pose is not None:
            # Calculate relative rotation from neutral pose using quaternions
            neutral_rot = R.from_matrix(self.vr_neutral_pose[:3, :3])
            current_rot = R.from_matrix(rot_mat[:3, :3])
            
            # Get relative rotation as quaternion
            relative_rot = neutral_rot.inv() * current_rot
            
            # Convert to axis-angle to apply transformation
            rotvec = relative_rot.as_rotvec()
            angle = np.linalg.norm(rotvec)
            
            if angle > 0:
                axis = rotvec / angle  # Normalize to get unit axis
                
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
                transformed_rotvec = transformed_axis * angle
                transformed_rot = R.from_rotvec(transformed_rotvec)
                vr_quat = transformed_rot.as_quat()
            else:
                # No rotation
                vr_quat = np.array([0, 0, 0, 1])
        else:
            # No neutral pose yet, apply standard transformation
            transformed_rot_mat = self.global_to_env_mat[:3, :3] @ self.vr_to_global_mat[:3, :3] @ rot_mat[:3, :3]
            vr_quat = rmat_to_quat(transformed_rot_mat)
        
        vr_gripper = self._state["buttons"]["rightTrig" if self.controller_id == "r" else "leftTrig"][0]

        self.vr_state = {"pos": vr_pos, "quat": vr_quat, "gripper": vr_gripper, "euler": quat_to_euler(vr_quat)}

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
        # Read Sensor
        if self.update_sensor:
            self._process_reading()
            self.update_sensor = False
        
        # Check if we have valid data
        if self.vr_state is None or self.robot_pos is None:
            return np.zeros(7), {}
        
        # Reset Origin On Release
        if self.reset_origin:
            self.robot_origin = {"pos": self.robot_pos, "quat": self.robot_quat,"euler":self.robot_euler}
            self.vr_origin = {"pos": self.vr_state["pos"], "quat": self.vr_state["quat"],"euler":self.vr_state["euler"]}
            self.reset_origin = False
            print("📍 Origin calibrated")
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
            print(f"\n📍 Translation Debug:")
            print(f"   VR Origin: [{self.vr_origin['pos'][0]:.4f}, {self.vr_origin['pos'][1]:.4f}, {self.vr_origin['pos'][2]:.4f}]")
            print(f"   VR Current: [{self.vr_state['pos'][0]:.4f}, {self.vr_state['pos'][1]:.4f}, {self.vr_state['pos'][2]:.4f}]")
            print(f"   VR Offset: [{target_pos_offset[0]:.4f}, {target_pos_offset[1]:.4f}, {target_pos_offset[2]:.4f}]")
            print(f"   Robot Offset: [{robot_pos_offset[0]:.4f}, {robot_pos_offset[1]:.4f}, {robot_pos_offset[2]:.4f}]")
            print(f"   Pos Action: [{pos_action[0]:.4f}, {pos_action[1]:.4f}, {pos_action[2]:.4f}]")
            print(f"   Pos Action Norm: {np.linalg.norm(pos_action):.4f}")
        print(f"\n📍 Translation Debug:")
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
        print(f"\n📍 scale_euler_action Debug:{euler_action}")
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
            "movement_enabled": self._state["movement_enabled"],
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
            print("🔄 [DEBUG] Would reset robot to initial position")
            # Return simulated values
            return np.array([0.4, 0.0, 0.3]), np.array([1.0, 0.0, 0.0])
        
        print("🔄 Resetting robot to initial position...")
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
        print(f"✅ Robot reset complete")
        print(f"   Position: [{robot_pos[0]:.6f}, {robot_pos[1]:.6f}, {robot_pos[2]:.6f}]")
        print(f"  euler: [{robot_euler[0]:.6f}, {robot_euler[1]:.6f}, {robot_euler[2]:.6f}]")
        
     
        
        #返回 末端位姿,四元数,关节角,总之就是返回机器人当前位姿
       
        return robot_pos,robot_euler,robot_joint
    
    def velocity_to_position_target(self, velocity_action, current_pos, current_quat, action_info=None):
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
        
        # Calculate target position
        target_pos = current_pos + pos_delta
        
        # Calculate target orientation - CRITICAL DIFFERENCE FOR DEOXYS
        if action_info and "target_quaternion" in action_info:
            # Use the pre-calculated target quaternion for Deoxys
            target_quat = action_info["target_quaternion"]
        else:
            # Fallback: convert rotation velocity (euler angles) to quaternion
            # This is less accurate for Deoxys but maintains compatibility
            rot_delta_quat = euler_to_quat(rot_delta)
            current_rot = R.from_quat(current_quat)
            delta_rot = R.from_quat(rot_delta_quat)
            target_rot = delta_rot * current_rot
            target_quat = target_rot.as_quat()
        
        # Calculate target gripper
        target_gripper = np.clip(self.robot_gripper + gripper_delta, 0.0, 1.0)
        
        if self.debug:
            # Debug quaternion transformation
            current_euler = quat_to_euler(current_quat, degrees=True)
            target_euler = quat_to_euler(target_quat, degrees=True)
            print(f"\n🔄 Rotation Debug:")
            
            # VR Controller Data
            print(f"   === VR CONTROLLER ===")
            if hasattr(self, 'vr_state') and self.vr_state:
                vr_euler = quat_to_euler(self.vr_state['quat'], degrees=True)
                print(f"   VR Quat (transformed): [{self.vr_state['quat'][0]:.3f}, {self.vr_state['quat'][1]:.3f}, {self.vr_state['quat'][2]:.3f}, {self.vr_state['quat'][3]:.3f}]")
                print(f"   VR Euler (transformed): [R:{vr_euler[0]:6.1f}, P:{vr_euler[1]:6.1f}, Y:{vr_euler[2]:6.1f}]")
            
            # Show raw VR data if available
            if hasattr(self, '_state') and self._state.get('poses') and self.controller_id in self._state['poses']:
                raw_pose = np.asarray(self._state['poses'][self.controller_id])
                raw_rot = R.from_matrix(raw_pose[:3, :3])
                raw_quat = raw_rot.as_quat()
                raw_euler = raw_rot.as_euler('xyz', degrees=True)
                print(f"   VR Raw Quat: [{raw_quat[0]:.3f}, {raw_quat[1]:.3f}, {raw_quat[2]:.3f}, {raw_quat[3]:.3f}]")
                print(f"   VR Raw Euler: [R:{raw_euler[0]:6.1f}, P:{raw_euler[1]:6.1f}, Y:{raw_euler[2]:6.1f}]")
                
                # Show relative rotation if neutral pose exists
                if hasattr(self, 'vr_neutral_pose') and self.vr_neutral_pose is not None:
                    neutral_rot = R.from_matrix(self.vr_neutral_pose[:3, :3])
                    relative_rot = neutral_rot.inv() * raw_rot
                    rotvec = relative_rot.as_rotvec()
                    angle = np.linalg.norm(rotvec)
                    if angle > 0:
                        axis_norm = rotvec / angle
                        print(f"   VR Rotation from neutral: {np.degrees(angle):.1f}° around [{axis_norm[0]:.2f}, {axis_norm[1]:.2f}, {axis_norm[2]:.2f}]")
                    
                    # Also show euler angle changes from neutral
                    neutral_euler = neutral_rot.as_euler('xyz', degrees=True)
                    euler_diff = raw_euler - neutral_euler
                    print(f"   VR Euler change from neutral: [ΔR:{euler_diff[0]:6.1f}, ΔP:{euler_diff[1]:6.1f}, ΔY:{euler_diff[2]:6.1f}]")
            
            print(f"   === ROBOT ===")
            print(f"   Robot Current Euler: [R:{current_euler[0]:6.1f}, P:{current_euler[1]:6.1f}, Y:{current_euler[2]:6.1f}]")
            print(f"   Robot Rot Delta (deg): [R:{np.degrees(rot_delta[0]):6.1f}, P:{np.degrees(rot_delta[1]):6.1f}, Y:{np.degrees(rot_delta[2]):6.1f}]")
            print(f"   Robot Target Euler: [R:{target_euler[0]:6.1f}, P:{target_euler[1]:6.1f}, Y:{target_euler[2]:6.1f}]")
            print(f"   Robot Current Quat: [{current_quat[0]:.3f}, {current_quat[1]:.3f}, {current_quat[2]:.3f}, {current_quat[3]:.3f}]")
            print(f"   Robot Target Quat: [{target_quat[0]:.3f}, {target_quat[1]:.3f}, {target_quat[2]:.3f}, {target_quat[3]:.3f}]")
        
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
            init_pos, init_euler,init_joint_positions = self.reset_robot()
            print(f" init_euler: {init_euler}")
            self.robot_pos = init_pos
            self.robot_quat = euler_to_quat(init_euler)
            self.robot_euler = init_euler
            self.robot_gripper = 0.0
            self.robot_joint_positions = init_joint_positions
            self.is_first_frame = False
            
            # Store initial robot state
            with self._robot_state_lock:
                self._latest_robot_state = RobotState(
                    timestamp=time.time(),
                    pos=init_pos,
                    quat=self.robot_quat ,
                    euler=init_euler,
                    gripper=self.robot_gripper,
                    joint_positions=self.robot_joint_positions,
                )
        # Start camera manager if enabled
        if self.camera_manager:
            try:
                self.camera_manager.start()
                print("📷 Camera manager started")
            except Exception as e:
                print(f"⚠️  Failed to start camera manager: {e}")
                # Continue without cameras
                self.camera_manager = None
        
        # Start worker threads
        if self.enable_recording and self.data_recorder:
            self._mcap_writer_thread = threading.Thread(target=self._mcap_writer_worker)
            self._mcap_writer_thread.daemon = True
            self._mcap_writer_thread.start()
            self._threads.append(self._mcap_writer_thread)
            print("✅ MCAP writer thread started")
            
            # Start data recording thread
            self._data_recording_thread = threading.Thread(target=self._data_recording_worker)
            self._data_recording_thread.daemon = True
            self._data_recording_thread.start()
            self._threads.append(self._data_recording_thread)
            print("✅ Data recording thread started")
        # if not self.debug:
        #     self._robot_comm_thread = threading.Thread(target=self._robot_comm_worker)
        #     self._robot_comm_thread.daemon = True
        #     self._robot_comm_thread.start()
        #     self._threads.append(self._robot_comm_thread)
        #     print("✅ Robot communication thread started")
        
        self._robot_control_thread = threading.Thread(target=self._robot_control_worker)
        self._robot_control_thread.daemon = True
        self._robot_control_thread.start()
        self._threads.append(self._robot_control_thread)
        print("✅ Robot control thread started")
        
        # Main loop now status and handles high-level control
        while self.running:
            try:
                current_time = time.time()
                
                # Handle robot reset after calibration
                if hasattr(self, '_reset_robot_after_calibration') and self._reset_robot_after_calibration:
                    self._reset_robot_after_calibration = False
                    print("🤖 Executing robot reset after calibration...")
                    
                    # Pause control thread during reset
                    self._control_paused = True
                    time.sleep(0.1)  # Give control thread time to pause
                    
                    reset_pos, reset_euler,reset_joints= self.reset_robot()
                    
                    # Update robot state
                    with self._robot_state_lock:
                        self._latest_robot_state = RobotState(
                            timestamp=current_time,
                            pos=reset_pos,
                            quat=euler_to_quat(reset_euler),
                            euler=reset_euler,
                            gripper=0.0,
                            joint_positions=reset_joints
                        )
                    
                    self.robot_pos = reset_pos
                    self.robot_quat = euler_to_quat(reset_euler)
                    self.robot_euler =reset_euler
                    self.robot_gripper = 0.0
                    # self.robot_joint_positions = reset_joint_positions
                    
                    # Clear any pending actions
                    self.reset_origin = True
                    
                    # Resume control thread
                    self._control_paused = False
                    
                    print("✅ Robot is now at home position, ready for teleoperation")
                    print("   Hold grip button to start controlling the robot\n")
                
                # Debug output
                if self.debug and current_time - last_debug_time > 1.0:
                    with self._vr_state_lock:
                        vr_state = self._latest_vr_state
                    with self._robot_state_lock:
                        robot_state = self._latest_robot_state
                    
                    if vr_state and robot_state:
                        print(f"\n📊 Async Status [{message_count:04d}]:")
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
                    print(f"❌ Error in main loop: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)
    
    def start(self):
        """Start the server"""
        try:
            # Run control loop
            self.control_loop()
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received")
            self.stop_server()
    
    def stop_server(self):
        """Gracefully stop the server"""
       
        if not self.running:
            return
            
        print("🛑 Stopping Oculus VR Server...")
        self.running = False
        
        # Stop any active recording
        if self.recording_active and self.data_recorder:
            print("📹 Stopping active recording...")
            self.data_recorder.stop_recording(success=False)
            self.recording_active = False
        # Send poison pill to MCAP writer
        if self._mcap_writer_thread and self._mcap_writer_thread.is_alive():
            self.mcap_queue.put(None)
        
        # Send poison pill to robot comm thread
        if self._robot_comm_thread and self._robot_comm_thread.is_alive():
            self._robot_command_queue.put(None)
        
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
                print("✅ Oculus Reader stopped")
            except Exception as e:
                print(f"⚠️  Error stopping Oculus Reader: {e}")
        
        # Stop camera manager
        if self.camera_manager:
            try:
                self.camera_manager.stop()
                print("✅ Camera manager stopped")
            except Exception as e:
                print(f"⚠️  Error stopping camera manager: {e}")
        
        # Stop simulation server if running
        if self.sim_server:
            self.sim_server.stop()
            print("✅ Simulation server stopped")
        
        # Close robot connections
        if not self.debug:
            if hasattr(self, 'action_socket'):
                self.action_socket.close()
            if hasattr(self, 'controller_publisher'):
                self.controller_publisher.close()
            if hasattr(self, 'context'):
                self.context.term()
            print("✅ Robot connections and ZMQ resources closed")
        
        print("✅ Server stopped gracefully")
        sys.exit(0)

    def _mcap_writer_worker(self):
        """Asynchronous MCAP writer thread - processes data without blocking control"""
        print("📹 MCAP writer thread started")
        
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
                print(f"❌ Error in MCAP writer: {e}")
                import traceback
                traceback.print_exc()
        
        print("📹 MCAP writer thread stopped")

    def _robot_control_worker(self):
        """Asynchronous robot control thread - sends commands at control frequency"""
        print("🤖 Robot control thread started")
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
                
                # Skip if control is paused
                if self._control_paused:
                    time.sleep(0.01)
                    continue
                
                # Control at specified frequency
                if current_time - last_control_time >= self.control_interval:
                    # Get latest VR state
                    with self._vr_state_lock:
                        vr_state = self._latest_vr_state.copy() if self._latest_vr_state else None
                    
                    # Get latest robot state
                    with self._robot_state_lock:
                        robot_state = self._latest_robot_state.copy() if self._latest_robot_state else None
                    
                    if vr_state and robot_state:
                        # Check if we're using predictions
                        state_age = current_time - robot_state.timestamp
                        if state_age > self.control_interval * 2:
                            if not using_predictions:
                                using_predictions = True
                        else:
                            if using_predictions:
                                using_predictions = False
                        
                        # Process control command
                        self._process_control_cycle(vr_state, robot_state, current_time)
                        control_count += 1
                    
                    last_control_time = current_time
                    
                    # Print actual frequency every second
                    if current_time - freq_check_time >= 1.0:
                        actual_freq = control_count / (current_time - freq_check_time)
                        # if self.recording_active:
                        #     status = "PREDICTIVE" if using_predictions else "REAL-TIME"
                        #     print(f"⚡ Control frequency: {actual_freq:.1f}Hz (target: {self.control_hz}Hz) - {status}")
                        control_count = 0
                        freq_check_time = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
            except Exception as e:
                if self.running:
                    print(f"❌ Error in robot control: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
        
        print("🤖 Robot control thread stopped")

    def _data_recording_worker(self):
        """Records data at target frequency independent of robot control"""
        print("📊 Data recording thread started")
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
                                print("⚠️  MCAP queue full, dropping frame")
                    
                    last_record_time = current_time
                    
                    # Print actual frequency every second
                    if current_time - freq_check_time >= 1.0:
                        if self.recording_active:
                            actual_freq = record_count / (current_time - freq_check_time)
                            print(f"📊 Recording frequency: {actual_freq:.1f}Hz (target: {self.recording_hz}Hz)")
                        record_count = 0
                        freq_check_time = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
            except Exception as e:
                if self.running:
                    print(f"❌ Error in data recording: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
        
        print("📊 Data recording thread stopped")

    def _robot_comm_worker(self):
        """Handles robot communication asynchronously to prevent blocking control thread"""
        print("🔌 Robot communication thread started")
        
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
                    print(f"📡 Avg robot comm: {avg_comm_time*1000:.1f}ms")
                
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
                    print(f"❌ Error in robot communication: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
        
        print("🔌 Robot communication thread stopped")

    def _process_control_cycle(self, vr_state: VRState, robot_state: RobotState, current_time: float):
        """Process a single control cycle with given VR and robot states"""
        # Restore state from thread-safe structures
        #self._state？
        self._state["poses"] = vr_state.poses
        self._state["buttons"] = vr_state.buttons
        self._state["movement_enabled"] = vr_state.movement_enabled
        self._state["controller_on"] = vr_state.controller_on
      
        # self._state["grip_released"] = vr_state.grip_released
        # self._state["toggle_state"] = vr_state.toggle_state
        # print(f"    grip_released state: {self._state['grip_released']}")
        # Update robot state
        self.robot_pos = robot_state.pos
        self.robot_quat = robot_state.quat
        self.robot_euler = robot_state.euler
        self.robot_gripper = robot_state.gripper
        self.robot_joint_positions = robot_state.joint_positions
        
        # Get controller info
        info = self.get_info()
        
        if self.enable_recording and self.data_recorder:
            # A button: Start/Reset recording
            current_a_button = info["success"]
            if current_a_button and not self.prev_a_button:  # Rising edge
                if self.recording_active:
                    # Stop current recording
                    print("\n🛑 A button pressed - Stopping current recording...")
                    self.data_recorder.reset_recording()
                    self.recording_active = False
                    print("📹 Recording stopped (not saved)")
                    print("   Press A to start a new recording")
                else:
                    # Start recording
                    print("\n▶️  A button pressed - Starting recording...")
                    self.data_recorder.start_recording()
                    self.recording_active = True
                    print("📹 Recording started")
                    print("   Press A again to stop/discard")
                    print("   Press B to mark as successful and save")
            self.prev_a_button = current_a_button
            
            # B button: Mark as successful and stop
            if info["failure"] and self.recording_active:
                print("\n✅ B button pressed - Marking recording as successful...")
                saved_filepath = self.data_recorder.stop_recording(success=True)
                self.recording_active = False
                print("📹 Recording saved successfully")
                
                # Verify data if requested
                if self.verify_data and saved_filepath:
                    print("\n🔍 Verifying recorded data...")
                    try:
                        verifier = MCAPVerifier(saved_filepath)
                        results = verifier.verify(verbose=True)
                        
                        # Check if verification passed
                        if not results["summary"]["is_valid"]:
                            print("\n⚠️  WARNING: Data verification found issues!")
                            print("   The recording may not be suitable for training.")
                    except Exception as e:
                        print(f"\n❌ Error during verification: {e}")
                        import traceback
                        traceback.print_exc()
                
                print("\n   Press A to start a new recording")
        else:
            # Original behavior when recording is disabled
            if info["success"]:
                print("\n✅ Success button pressed!")
                if not self.debug:
                    # Send termination signal
                    self.stop_server()
                    return
            
            if info["failure"]:
                print("\n❌ Failure button pressed!")
                if not self.debug:
                    # Send termination signal
                    self.stop_server()
                    return
        
        # Default action (no movement)
        action = np.zeros(7)
        action_info = {}
        if  self._state["toggle_state"]:
            if self._robot_gripper_count==0:
                print("    Gripper opened")
                self._robot.set_gripper_params(position=1000, force=30, speed=50,block=0)
            else:
                print("    Gripper closed")
                self._robot.grasp(timeout=8)
            self._state["toggle_state"]=False

        #独立控制rz
        joyvalue=self._state["buttons"].get("rightJS" if self.right_controller else "leftJS", [0.0])[0]
        # print("    joyvalue:",joyvalue)
        if abs(joyvalue)>0.5:    
            if joyvalue>0:          rz_delta=np.asarray([0,0,0,0,0,radians(0.5)])
            else:                   rz_delta=np.asarray([0,0,0,0,0,-radians(0.5)])
            self._robot.move_linear_extend(rz_delta, _speed=200, _accel=100,TOL=0.1,_is_block=False,_move_mode=1)
        # # Calculate action if movement is enabled
        if info["movement_enabled"] and self._state["poses"]:
            #vr位姿更新并且运动使能
            self._robot.robot.servo_move_enable(True)
            action, action_info = self._calculate_action()
            #output:末端线速度角速度和夹爪速度
            # Store last action for recording thread
            self._last_action = action.copy()
            
            # Convert velocity to position target
            target_pos, target_euler, target_gripper = self.velocity_to_position_target(
                action, self.robot_pos, self.robot_quat, action_info
            )
            pos_scale =1
            euler_scale = 1
           
            delta=np.asarray([target_pos[0]*1000*pos_scale,target_pos[1]*1000*pos_scale,target_pos[2]*1000*pos_scale,-target_euler[0]*euler_scale,-target_euler[1]*euler_scale,target_euler[2]*euler_scale])
            # delta=np.asarray([target_pos[0]*1000*pos_scale,target_pos[1]*1000*pos_scale,target_pos[2]*1000*pos_scale,0,-target_euler[1]*euler_scale,0   ])#RY
            # delta=np.asarray([target_pos[0]*1000*pos_scale,target_pos[1]*1000*pos_scale,target_pos[2]*1000*pos_scale,-target_euler[0]*euler_scale,0,0   ])#RX
            # delta=np.asarray([target_pos[0]*1000*pos_scale,target_pos[1]*1000*pos_scale,target_pos[2]*1000*pos_scale,0,0,target_euler[2]*euler_scale   ])#RZ
            delta_degree=np.asarray([target_pos[0]*1000*pos_scale,target_pos[1]*1000*pos_scale,target_pos[2]*1000*pos_scale,np.degrees(target_euler[0]*euler_scale),np.degrees(target_euler[1]*euler_scale),np.degrees(target_euler[2]*euler_scale)])
            
            print(f"    delta_degree:{delta_degree}")
            tar_delta=np.asarray([target_pos[0]*pos_scale,target_pos[1]*pos_scale,target_pos[2]*pos_scale  , radians(0.0), radians(0.0), radians(0)])
          


            # Send action to robot (or simulate)
            if not self.debug:
                # Send action to robot - DEOXYS EXPECTS QUATERNIONS
                _step_num=5
                if self.enable_performance_mode:_step_num=2
                
                try:
                    # pass
                    self._robot.robot.servo_p_extend(delta, move_mode=1, step_num=_step_num)
                except Exception as e:
                    print(f"    ❌ Error in robot move: {e}")
                
                try:
                    new_robot_cartesian=np.array(self._robot.current_cartesian)
                    new_robot_state=np.array(self._robot.get_current_TCP()[1])
                    new_robot_joints=np.array(self._robot.current_joints)
                    print(f"    new_robot_state:{new_robot_state}")
                    print(f"    new_robot_cartesian:{new_robot_cartesian}")
                    print(f"    new_robot_pos:{new_robot_cartesian[:3]}")
                    print(f"    new_robot_euler:{np.degrees(new_robot_cartesian[3:])}")
                    new_robot_state = RobotState(
                        timestamp=current_time,
                        pos=0.001*new_robot_cartesian[:3],
                        quat=euler_to_quat(new_robot_cartesian[3:]),
                        euler=new_robot_cartesian[3:],
                        gripper= self._robot_gripper_count,
                        joint_positions=new_robot_joints        
                    )
                    with self._robot_state_lock:
                        self._latest_robot_state = new_robot_state
                    #获取线程锁更新机器人状态

                    # Update local state for next calculation
                    self.robot_pos = new_robot_state.pos
                    self.robot_quat = new_robot_state.quat
                    self.robot_euler = new_robot_state.euler
                    self.robot_gripper = new_robot_state.gripper
                    self.robot_joint_positions = new_robot_state.joint_positions
                except Exception as e:
                    print(f"    ❌ Error in robot_state update: {e}")
        else:
            # Not moving - use current robot state
            new_robot_state = robot_state

            # Clear last action when not moving
            self._last_action = np.zeros(7)
            if not self.debug and self._state["grip_released"]:
                print("    Gripper released  Stop!")
               
                self._robot.robot.servo_move_enable(False)
                self._robot.stop_motion()
                self._state["grip_released"]=False
        
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
        
    #     print("🔥 Launching in hot reload mode...")
        
    #     # Check if hot reload script exists
    #     if not os.path.exists('oculus_vr_server_hotreload.py'):
    #         print("❌ Hot reload script not found!")
    #         print("   Make sure oculus_vr_server_hotreload.py is in the same directory")
    #         sys.exit(1)
        
    #     # Launch the hot reload wrapper
    #     try:
    #         subprocess.run([sys.executable, 'oculus_vr_server_hotreload.py'] + new_args)
    #     except KeyboardInterrupt:
    #         print("\n✅ Hot reload stopped")
    #     sys.exit(0)
    
    # # Handle auto-discovery of cameras
    # if args.auto_discover_cameras:
    #     print("🔍 Auto-discovering cameras...")
    #     try:
    #         from frankateach.camera_utils import discover_all_cameras, generate_camera_config
            
    #         cameras = discover_all_cameras()
    #         if cameras:
    #             # Generate temporary config
    #             temp_config = "/tmp/cameras_autodiscovered.yaml"
    #             generate_camera_config(cameras, temp_config)
                
    #             # Override camera config path
    #             args.camera_config = temp_config
    #             args.enable_cameras = True
                
    #             print(f"✅ Using auto-discovered cameras from: {temp_config}")
    #         else:
    #             print("⚠️  No cameras found during auto-discovery")
    #     except Exception as e:
    #         print(f"❌ Camera auto-discovery failed: {e}")
    
    # # Load camera configuration if provided
    # camera_configs = None
    # if args.camera_config:
    #     try:
    #         import yaml
    #         with open(args.camera_config, 'r') as f:
    #             camera_configs = yaml.safe_load(f)
    #         print(f"📷 Loaded camera configuration from {args.camera_config}")
    #     except Exception as e:
    #         print(f"⚠️  Failed to load camera config: {e}")
    #         print("   Continuing without camera configuration")
    
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
        camera_config_path=args.camera_config,
        enable_cameras=args.enable_cameras
    )
    try:
        server.start()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()   
        server.stop_server()
        server._robot.robot.servo_move_enable(False)


if __name__ == "__main__":
    main() 