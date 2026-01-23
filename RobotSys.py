# 单臂抓取脚本（坐标输入简化版）
# -*- coding: utf-8 -*-  
import sys  
import os
import time        
import cv2
import pyrealsense2 as rs
import numpy as np
from scipy.spatial.transform import Rotation as R
from math import radians
from camera_point import RealSense3DCoord

sys.path.append('./Modules')
# 正确导入所需模块
from Modules.jaka_control import *

# ---------------- 用户参数 ----------------
ROBOT_IP = "192.168.1.24"  # JAKA 控制器 IP

# 相机内外参（RGB 相机，和深度图对齐后使用同一套内参）
# CAMERA_INTRINSICS = np.array([[628.77969479, 0.0, 662.10508256],
#                                       [0.0, 627.99375281, 349.23865973 ],
#                                       [0.0, 0.0, 1.0]])
# CAMERA_DIST_COEFFS = np.array([-0.06656697, 0.05817578, -0.00485461, 0.00338863, -0.02526062])
# rotation_matrix = np.array([[0.99973908,-0.01340421,0.01849603],
#                             [-0.02228943 , -0.74954646 , 0.66157636],
#                             [0.00499572, -0.66181601, -0.74964966]])

# translation_vector = np.array([-0.23325556, -0.92798298 , 0.54279094])#单位米

CAMERA_INTRINSICS = np.array([[651.55099836,   0. ,        658.17448505],
                             [  0.,         650.9236726 , 369.37938845],
                                [  0. ,          0. ,          1.        ]])
CAMERA_DIST_COEFFS = np.array([[-0.06883659,  0.09069593 ,-0.00116802, -0.00314408 ,-0.04680663]])
rotation_matrix = np.array( [[ 0.99993038, -0.01118613 , 0.00375678],
                            [-0.01160588 ,-0.87477041 , 0.48439862],
                            [-0.00213223 ,-0.4844085,  -0.87483933]])

translation_vector = np.array([-0.09322482, -0.95651948 , 0.53782021 ])#单位米

# 机器人默认运动参数
SPEED = 500     # mm/s
ACC   = 200      # mm/s²
TOL   = 1     # mm
LOEST = -26     # 最低高度

# 安全位置
SAFE_POSITION = [-85, -350, 430, radians(180.0), radians(0.0), radians(45.0)]
PLACE_POSITION = [-467.67, -142, 248.8, radians(180.0), radians(-0.0), radians(122.0)]

class RobotSys:
    """
    机器人系统类，封装机器人初始化、相机控制和技能库功能
    """
    
    def __init__(self, robot_ip=ROBOT_IP):
        """
        初始化机器人系统
        Args:
            robot_ip: 机器人控制器IP地址
        """
        self.robot_ip = robot_ip
        self.robot = None
        self.camera = None
        self.is_initialized = False
        
        # 运动参数
        self.speed = SPEED
        self.acc = ACC
        self.tol = TOL
        
        # 安全位置
        self.safe_position = SAFE_POSITION
        self.place_position = PLACE_POSITION
        
        # 手眼标定参数
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        
        print(f"RobotSys 初始化完成，目标IP: {self.robot_ip}")
    
    def initialize(self, camera_intrinsics=None, dist_coeffs=None):
        """
        初始化机器人、相机和夹爪
        Args:
            camera_intrinsics: 相机内参矩阵，为None则自动获取
            dist_coeffs: 相机畸变系数，为None则自动获取
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 1. 初始化机器人
            print("正在初始化机器人...")
            self.robot = JakaRobot(self.robot_ip)
            success, message = self.robot.connect()
            if not success:
                print(f"机器人连接失败: {message}")
                return False
            
            self.robot.power_on()
            self.robot.enable()
            self.robot.robot.set_tool_id(1)
            self.robot.set_tool_data(140)  # 设置工具坐标系
            
            # 2. 初始化相机
            print("正在初始化相机...")
            self.camera = RealSense3DCoord(
                camera_intrinsics=camera_intrinsics or CAMERA_INTRINSICS,
                dist_coeffs=None,
                depth_width=848,
                depth_height=480,
                fps=30
            )
            
            if not self.camera.initialize():
                print("相机初始化失败")
                return False
            
            # 3. 初始化夹爪
            print("正在初始化夹爪...")
            self.robot.init_gripper(0x01)
            
            # 4. 移动到安全位置
            # print("移动到安全位置...")
            # if not self._move_to_safe_position():
            #     print("移动到安全位置失败")
            #     return False
            
            self.is_initialized = True
            print("机器人系统初始化完成！")
            return True
            
        except Exception as e:
            print(f"初始化过程中发生异常: {str(e)}")
            return False
    
    def shutdown(self):
        """
        关闭机器人系统
        """
        try:
            if self.robot and self.robot.is_connected:
                # 移动到安全位置
                self._move_to_safe_position()
                
                # 断开连接
                self.robot.disconnect()
            
            if self.camera:
                self.camera.shutdown()
                
            self.is_initialized = False
            print("机器人系统已关闭")
            
        except Exception as e:
            print(f"关闭过程中发生异常: {str(e)}")
    
    def _move_to_safe_position(self):
        """
        移动到安全位置（内部方法）
        """
        return self._move_linear(self.safe_position, "安全位置")
    
    def _move_linear(self, target_pos, desc="目标位置",block=1):
        """
        直线运动封装（内部方法）
        """
        try:
            ret = self.robot.robot.linear_move_extend(
                end_pos=target_pos,
                move_mode=0,
                is_block=block,
                speed=self.speed,
                acc=self.acc,
                tol=self.tol
            )
            if ret == (0,):
                print(f"已移动到{desc}：{target_pos}")
                return True
            else:
                print(f"移动到{desc}失败，返回值：{ret}")
                return False
        except Exception as e:
            print(f"移动过程中发生异常: {str(e)}")
            return False
    
    def _convert_coordinates(self, x, y, z):
        """
        坐标转换：相机坐标系 → 世界坐标系（内部方法）
        """
        # 深度相机识别物体返回的坐标
        obj_camera_coordinates = np.array([x, y, z])

        # 将旋转矩阵和平移向量转换为齐次变换矩阵
        T_camera_to_base_effector = np.eye(4)
        T_camera_to_base_effector[:3, :3] = self.rotation_matrix
        T_camera_to_base_effector[:3, 3] = self.translation_vector

        # 计算物体相对于机械臂基座的位姿
        obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1.0])  # 将物体坐标转换为齐次坐标
        obj_base_effector_coordinates_homo = T_camera_to_base_effector.dot(obj_camera_coordinates_homo)
        obj_base_coordinates = obj_base_effector_coordinates_homo[:3]  # 从齐次坐标中提取物体的x, y, z坐标

        return np.array(obj_base_coordinates)
    
    # -------------------- 技能库 --------------------
    def pick(self, target_position=None):
        """
        执行抓取任务
        Args:
            u: 像素x坐标
            v: 像素y坐标
        Returns:
            dict: 任务状态字典，包含success(布尔值)和message(字符串描述)
        """
        if not self.is_initialized:
            return {
                "success": False,
                "message": "机器人系统未初始化"
            }
        
        try:
            # 1. 获取相机坐标系下的3D坐标
            coord = self.camera.get_3d_coordinate(target_position[0], target_position[1])
            if coord is None:
                return {
                    "success": False,
                    "message": f"无法获取像素坐标({target_position})的3D坐标"
                }
            
            camera_pt_m = coord
            print(f"相机系坐标: {camera_pt_m}")

            # 2. 获取当前 TCP 位姿
            ret = self.robot.robot.get_tcp_position()
            if ret[0] != 0:
                return {
                    "success": False,
                    "message": f"获取 TCP 位姿失败，错误码：{ret[0]}"
                }

            tcp_pose = ret[1]
            tcp_pos = tcp_pose[:3]
            tcp_rpy = tcp_pose[3:6]
            
            print("TCP 位置 (mm):", tcp_pos)
            print("TCP 姿态 (rad):", tcp_rpy)

            # 3. 相机坐标系 → 世界坐标系
            target_world_m = self._convert_coordinates(camera_pt_m[0], camera_pt_m[1], camera_pt_m[2])
            target_world_mm = target_world_m * 1000.0
            print(f"目标点在世界坐标系的坐标（mm）: {target_world_mm}")
            
            # 计算中间过渡点和最终目标点
            target_world_mm_mid = target_world_mm + np.array([0, 0, 100])
            target_world_mm_final = target_world_mm 
            
            print(f"过渡坐标（mm）: {target_world_mm_mid}")

            # 4. 构造末端位姿（保持当前姿态，仅改变位置）
            end_pos_mid = list(target_world_mm_mid) + list(tcp_rpy)
            end_pos_final = list(target_world_mm_final) + list(tcp_rpy)
            
            print(f"最终目标位姿: {end_pos_final}")

            # 5. 移动到目标位置
            print(f"先移动到上方位置")
            if not self._move_linear(end_pos_mid, "上方过渡位置"):
                return {
                    "success": False,
                    "message": "移动到上方过渡位置失败"
                }
            
            if not self._move_linear(end_pos_final, "抓取位置"):
                return {
                    "success": False,
                    "message": "移动到抓取位置失败"
                }
            
            # 6. 执行抓取
            success, message = self.robot.grasp(timeout=8.0)
            if not success:
                return {
                    "success": False,
                    "message": f"抓取失败: {message}"
                }
            
            print("已夹取物体")
            time.sleep(1)
            
            # # 7. 抬起到安全高度
            # if not self._move_linear(end_pos_mid, "抬升位置"):
            #     return {
            #         "success": False,
            #         "message": "抬升失败"
            #     }

            print("抓取任务完成！")
            return {
                "success": True,
                "message": "抓取任务成功完成",
                "target_world_coordinates": target_world_mm.tolist()
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"执行过程中发生异常: {str(e)}"
            }
    
    def place(self, target_position=None):
        """
        放置物体到指定位置
        Args:
            target_position: 目标位置，为None则使用默认放置位置
        Returns:
            dict: 任务状态字典
        """
        
        if not self.is_initialized:
            return {
                "success": False,
                "message": "机器人系统未初始化"
            }
        
        try:
            # 1. 获取相机坐标系下的3D坐标
            coord = self.camera.get_3d_coordinate(target_position[0], target_position[1])
            if coord is None:
                return {
                    "success": False,
                    "message": f"无法获取像素坐标({target_position})的3D坐标"
                }
            
            camera_pt_m = coord
            print(f"相机系坐标: {camera_pt_m}")

            # 2. 获取当前 TCP 位姿
            ret = self.robot.robot.get_tcp_position()
            if ret[0] != 0:
                return {
                    "success": False,
                    "message": f"获取 TCP 位姿失败，错误码：{ret[0]}"
                }

            tcp_pose = ret[1]
            tcp_pos = tcp_pose[:3]
            tcp_rpy = tcp_pose[3:6]
            
            print("TCP 位置 (mm):", tcp_pos)
            print("TCP 姿态 (rad):", tcp_rpy)

            # 3. 相机坐标系 → 世界坐标系
            target_world_m = self._convert_coordinates(camera_pt_m[0], camera_pt_m[1], camera_pt_m[2])
            target_world_mm = target_world_m * 1000.0
            print(f"目标点在世界坐标系的坐标（mm）: {target_world_mm}")
            
            # 计算中间过渡点和最终目标点
            target_world_mm_mid = target_world_mm + np.array([0, 0, 300])
            target_world_mm_final = target_world_mm + np.array([0, 0, 150])
            target_world_mm_first= tcp_pos+ np.array([0, 0, 300])
            print(f"过渡坐标（mm）: {target_world_mm_mid}")

            # 4. 构造末端位姿（保持当前姿态，仅改变位置）
            end_pos_first = list(target_world_mm_first) + list(tcp_rpy)
            end_pos_mid = list(target_world_mm_mid) + list(tcp_rpy)
            end_pos_final = list(target_world_mm_final) + list(tcp_rpy)
            
            print(f"最终目标位姿: {end_pos_final}")

            # 5. 移动到目标位置
            print(f"先移动到上方位置")
            self._move_linear(end_pos_first, "上方过渡位置")
            self._move_linear(end_pos_mid, "上方过渡位置")
                
            
            self._move_linear(end_pos_final, "放置位置")
             
            # 6. 执行放置
            success, message =  self.robot.set_gripper_params(position=1000, force=30, speed=50,block=1)
            
         
      

            print("放置任务完成！")
            return {
                "success": True,
                "message": "放置任务完成！",
                "target_world_coordinates": target_world_mm.tolist()
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"执行过程中发生异常: {str(e)}"
            }
    
    def get_camera_frame(self):
        """
        获取当前相机帧
        Returns:
            tuple: (color_frame, depth_frame) 或 (None, None) 如果获取失败
        """
        if not self.is_initialized or not self.camera:
            return None, None
        
        try:
            # 这里需要根据你的RealSense3DCoord类的实际接口调整
            # 假设你的类有获取帧的方法
            color_frame = self.camera.get_color_frame()
            depth_frame = self.camera.get_depth_frame()
            return color_frame, depth_frame
        except Exception as e:
            print(f"获取相机帧失败: {str(e)}")
            return None, None
    
    def set_motion_parameters(self, speed=None, acc=None, tol=None):
        """
        设置运动参数
        Args:
            speed: 速度(mm/s)
            acc: 加速度(mm/s²)
            tol: 容差(mm)
        """
        if speed is not None:
            self.speed = speed
        if acc is not None:
            self.acc = acc
        if tol is not None:
            self.tol = tol
        
        print(f"运动参数已更新: 速度={self.speed}, 加速度={self.acc}, 容差={self.tol}")

# -------------------- 使用示例 --------------------
if __name__ == "__main__":
    # 创建机器人系统实例
    robot_sys = RobotSys(ROBOT_IP)
    
    try:
        # 初始化系统
        robot_sys.initialize()
        print("系统初始化成功，开始执行任务...")
            
        robot_sys._move_to_safe_position()
         
        result = robot_sys.pick(686,551) 
        print("抓取任务结果:", result["message"])
            
            
     
      
    
    except Exception as e:
        print(f"主程序执行异常: {str(e)}")
    
    finally:
        # 关闭系统
        robot_sys.shutdown()