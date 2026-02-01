#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys  
import os
import numpy as np
import time
import jkrc    
import threading
import time
import logging
from math import degrees, radians
FORCE=50
SPEED=70
'''
注意python版本过高会导致jkrc无法解析


'''
# -------------------- 日志配置 --------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("JakaRobot")

class GripperCommandBuilder:
    """夹爪命令构建器 - 专门负责构建Modbus-RTU命令帧"""
    
    def __init__(self, slave_id=0x01,name:str="gripper"):
        self.slave_id = slave_id  # 默认夹爪ID为1
        self.gripper_name=name #夹爪名称，用于日志区分

        self.gripper_monitoring = False
        self.current_gripper_position = 0
        self.current_gripper_force = 50
        self.current_gripper_speed = 50
        self.gripper_status_callbacks = []

        # 夹爪参数范围定义
        self.FORCE_MIN = 20    # 最小力值百分比
        self.FORCE_MAX = 100   # 最大力值百分比
        self.POSITION_MIN = 0# 最小位置千分比
        self.POSITION_MAX = 1000 # 最大位置千分比
        self.SPEED_MIN = 1     # 最小速度百分比
        self.SPEED_MAX = 100   # 最大速度百分比
        
        # Modbus寄存器地址定义（基于文档2.3.2节）
        self.REG_INIT = 0x0100      # 初始化夹爪
        self.REG_FORCE = 0x0101    # 力值设置
        self.REG_POSITION = 0x0103 # 位置设置
        self.REG_SPEED = 0x0104    # 速度设置
        self.REG_SAVE = 0x0300     # 保存配置

        logger.info("GripperCommandBuilder 初始化完成，从站ID: 0x%02X, 名称: %s", self.slave_id, self.gripper_name)

    def _calculate_crc(self, data):
        """计算Modbus-RTU CRC16校验码"""
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for i in range(8):
                if ((crc & 1) != 0):
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return [(crc & 0xFF), ((crc >> 8) & 0xFF)]

    def build_init_command(self, init_type=0x01):
        """
        构建夹爪初始化命令
        Args:
            init_type: 0x01 - 普通初始化, 0xA5 - 重新标定初始化
        Returns:
            bytearray: Modbus-RTU命令帧
        """
        if init_type not in [0x01, 0xA5]:
            raise ValueError("初始化类型错误，请使用0x01或0xA5")
        
        frame = [
            self.slave_id,           # 从站地址
            0x06,                    # 功能码：写入单个寄存器
            (self.REG_INIT >> 8) & 0xFF,  # 寄存器地址高字节
            self.REG_INIT & 0xFF,         # 寄存器地址低字节
            0x00,                    # 数据高字节（初始化类型为单字节）
            init_type               # 数据低字节：初始化类型
        ]
        
        # 计算CRC并添加到帧
        crc = self._calculate_crc(frame)
        frame.extend(crc)
        
        cmd_hex = ' '.join(f'{b:02X}' for b in frame)
        logger.info(f"构建初始化命令: {cmd_hex} (类型: 0x{init_type:02X})")
        
        return bytearray(frame)

    def build_force_command(self, force_value):
        """
        构建力值设置命令
        Args:
            force_value: 力值 (20-100 百分比)
        Returns:
            bytearray: Modbus-RTU命令帧
        """
        if not (self.FORCE_MIN <= force_value <= self.FORCE_MAX):
            raise ValueError(f"力值超出范围 ({self.FORCE_MIN}-{self.FORCE_MAX})")
        
        frame = [
            self.slave_id,
            0x06,
            (self.REG_FORCE >> 8) & 0xFF,
            self.REG_FORCE & 0xFF,
            (force_value >> 8) & 0xFF,
            force_value & 0xFF
        ]
        
        crc = self._calculate_crc(frame)
        frame.extend(crc)
        
        cmd_hex = ' '.join(f'{b:02X}' for b in frame)
        logger.info(f"构建力值命令: {cmd_hex} (力值: {force_value}%)")
        
        return bytearray(frame)

    def build_position_command(self, position_value):
        """
        构建位置设置命令
        Args:
            position_value: 位置 (0-1000 千分比)
        Returns:
            bytearray: Modbus-RTU命令帧
        """
        if not (self.POSITION_MIN <= position_value <= self.POSITION_MAX):
            raise ValueError(f"位置超出范围 ({self.POSITION_MIN}-{self.POSITION_MAX})")
        
        frame = [
            self.slave_id,
            0x06,
            (self.REG_POSITION >> 8) & 0xFF,
            self.REG_POSITION & 0xFF,
            (position_value >> 8) & 0xFF,
            position_value & 0xFF
        ]
        
        crc = self._calculate_crc(frame)
        frame.extend(crc)
        
        cmd_hex = ' '.join(f'{b:02X}' for b in frame)
        logger.info(f"构建位置命令: {cmd_hex} (位置: {position_value}‰)")
        
        return bytearray(frame)

    def build_speed_command(self, speed_value):
        """
        构建速度设置命令
        Args:
            speed_value: 速度 (1-100 百分比)
        Returns:
            bytearray: Modbus-RTU命令帧
        """
        if not (self.SPEED_MIN <= speed_value <= self.SPEED_MAX):
            raise ValueError(f"速度超出范围 ({self.SPEED_MIN}-{self.SPEED_MAX})")
        
        frame = [
            self.slave_id,
            0x06,
            (self.REG_SPEED >> 8) & 0xFF,
            self.REG_SPEED & 0xFF,
            (speed_value >> 8) & 0xFF,
            speed_value & 0xFF
        ]
        
        crc = self._calculate_crc(frame)
        frame.extend(crc)
        
        cmd_hex = ' '.join(f'{b:02X}' for b in frame)
        logger.info(f"构建速度命令: {cmd_hex} (速度: {speed_value}%)")
        
        return bytearray(frame)

    def build_save_command(self):
        """构建保存配置命令"""
        frame = [
            self.slave_id,
            0x06,
            (self.REG_SAVE >> 8) & 0xFF,
            self.REG_SAVE & 0xFF,
            0x00,  # 数据高字节
            0x01   # 数据低字节：保存指令
        ]
        
        crc = self._calculate_crc(frame)
        frame.extend(crc)
        
        cmd_hex = ' '.join(f'{b:02X}' for b in frame)
        logger.info(f"构建保存命令: {cmd_hex}")
        
        return bytearray(frame)

class JakaRobot:
    def __init__(self, ip_address, gripper_name: str = "gripper", gripper_slave_id: int = 0x01):
        self.ip = ip_address
        self.robot = None
        self.is_connected = False
        self.heartbeat_running = False
        self.heartbeat_thread = None
        self.heartbeat_interval=0.05
        
        self.gripper_monitoring = False
        self.current_gripper_position = 0
        self.current_gripper_force = 50
        self.current_gripper_speed = 50
        self.gripper_status_callbacks = []

        #夹爪标识（用于区分左右臂夹爪）
        self.gripper_name = gripper_name
        self.gripper_slave_id = gripper_slave_id

        # 夹爪命令构建器实例
        self.gripper_builder = GripperCommandBuilder(slave_id=self.gripper_slave_id, name=self.gripper_name)
      
        # 运动参数
        self.joint_speed = 20
        self.joint_accel = 4.0
        self.linear_speed = 10.0
        self.linear_accel = 40.0

        # 当前位置
        self.current_joints = [0.0] * 6
        self.current_cartesian = [0.0] * 6

        # 坐标系常量
        self.COORD_BASE = 0
        self.COORD_JOINT = 1
        self.COORD_TOOL = 2
         # 新增夹爪状态监测相关属性
        self.gripper_monitoring = False
        self.gripper_monitor_thread = None
        self.current_gripper_status = 0
        self.current_gripper_position = 0
        self.gripper_status_callbacks = []
        self.gripper_data_lock = threading.Lock()
        
        # 夹爪信号配置（可根据实际夹爪文档调整标识名）
        # self.gripper_signal_config = {
        #     'status_run': 'status_run',  # 状态变量标识名
        #     'position': 'position'       # 位置变量标识名
        # }
        self.gripper_signal_config = {
            'status_run': 513,  # 状态变量标识名
            'position': 514       # 位置变量标识名
        }
        # 信号刷新频率（Hz），不超过20Hz
        self.gripper_signal_frequency = 1
        logger.info("JakaRobot 实例已创建，目标 IP: %s", self.ip)

    def _send_gripper_command(self, command_frame):
        """
        发送夹爪命令到机器人控制器
        Args:
            command_frame: 由GripperCommandBuilder构建的命令帧
        Returns:
            tuple: (成功状态, 消息)
        """
        if not self.is_connected:
            return False, "机械臂未连接"
        
        try:
            hex_cmd = command_frame.hex().upper()
            logger.debug(f"发送夹爪命令: {hex_cmd}")
            
            # 使用机器人接口发送RS485命令
            # 注意：send_tio_rs_command的第一个参数是端口号，根据实际情况调整
            ret = self.robot.send_tio_rs_command(0x1, command_frame)
            
            if ret[0] == 0:
                logger.info("夹爪命令发送成功")
                return True, "命令发送成功"
            else:
                logger.error(f"夹爪命令发送失败，错误码: {ret[0]}")
                return False, f"发送失败，错误码: {ret[0]}"
                
        except Exception as e:
            logger.exception("夹爪命令发送异常")
            return False, f"发送异常: {str(e)}"

    # -------------------- 夹爪控制函数 --------------------
    def init_gripper(self, init_type=0x01):
        """
        夹爪初始化
        Args:
            init_type: 0x01 - 普通初始化, 0xA5 - 重新标定初始化
        """
        logger.info(f"开始夹爪初始化，类型: {init_type}")
        
        try:
            # 构建初始化命令
            command_frame = self.gripper_builder.build_init_command(init_type)
            
            # 发送命令
            success, message = self._send_gripper_command(command_frame)
            
            if success:
                # 等待初始化完成
                if init_type == 0xA5:
                    wait_time = 3.0  # 重新标定需要更长时间
                else:
                    wait_time = 3.5  # 普通初始化时间
                
                logger.info(f"夹爪初始化中，等待{wait_time}秒...")
                time.sleep(wait_time)
                logger.info("夹爪初始化完成")
                return True, "初始化完成"
            
            return success, message
            
        except ValueError as e:
            logger.error(f"初始化命令构建失败: {str(e)}")
            return False, f"初始化失败: {str(e)}"
        except Exception as e:
            logger.exception("夹爪初始化异常")
            return False, f"初始化异常: {str(e)}"

    def set_gripper_force(self, force_value):
        """
        设置夹爪力值
        Args:
            force_value: 力值 (20-100 百分比)
        """
        logger.info(f"设置夹爪力值: {force_value}%")
        
        try:
            command_frame = self.gripper_builder.build_force_command(force_value)
            ret = self._send_gripper_command(command_frame)
            #如果设置成功，更新当前力值
            if ret[0]:
                self.current_gripper_force = force_value
            return ret

        except ValueError as e:
            logger.error(f"力值命令构建失败: {str(e)}")
            return False, f"力值设置失败: {str(e)}"
        except Exception as e:
            logger.exception("力值设置异常")
            return False, f"力值设置异常: {str(e)}"

    def set_gripper_position(self, position_value):
        """
        设置夹爪位置
        Args:
            position_value: 位置 (0-1000 千分比)
        """
        logger.info(f"设置夹爪位置: {position_value}‰")
        
        try:
            command_frame = self.gripper_builder.build_position_command(position_value)
            return self._send_gripper_command(command_frame)
            
        except ValueError as e:
            logger.error(f"位置命令构建失败: {str(e)}")
            return False, f"位置设置失败: {str(e)}"
        except Exception as e:
            logger.exception("位置设置异常")
            return False, f"位置设置异常: {str(e)}"

    def set_gripper_speed(self, speed_value):
        """
        设置夹爪速度
        Args:
            speed_value: 速度 (1-100 百分比)
        """
        logger.info(f"设置夹爪速度: {speed_value}%")
        
        try:
            command_frame = self.gripper_builder.build_speed_command(speed_value)
            ret = self._send_gripper_command(command_frame)
            if ret[0]:
                self.current_gripper_speed = speed_value
            return ret

        except ValueError as e:
            logger.error(f"速度命令构建失败: {str(e)}")
            return False, f"速度设置失败: {str(e)}"
        except Exception as e:
            logger.exception("速度设置异常")
            return False, f"速度设置异常: {str(e)}"
    def set_gripper_params(self, position=None, force=FORCE, speed=SPEED, block=0, timeout=5.0, check_interval=0.1):
        """
        综合设置夹爪参数，并可选择是否阻塞等待夹爪就绪
        Args:
            position: 位置 (0-1000 千分比)
            force: 力值 (20-100 百分比)
            speed: 速度 (1-100 百分比)
            block: 阻塞模式 (0-非阻塞, 1-阻塞)
            timeout: 阻塞等待的超时时间(秒)
            check_interval: 状态检查间隔(秒)
        Returns:
            tuple: (成功状态, 消息)
        """
        logger.info(f"设置夹爪参数 - 位置: {position}, 力: {force}, 速度: {speed}, 阻塞模式: {block}, 超时: {timeout}秒")
        
        # 确保夹爪状态监测正在运行（阻塞模式下必需）
        if block == 1 and not self.gripper_monitoring:
            self.start_gripper_monitoring()
            time.sleep(0.1)  # 等待监测线程稳定
        
        results = []
        
        try:
            # 设置力值
            if force is not None:
                success, message = self.set_gripper_force(force)
                results.append(("力值", success, message))
                if not success:
                    return False, f"力值设置失败: {message}"

            # 设置位置
            if position is not None:
                success, message = self.set_gripper_position(position)
                results.append(("位置", success, message))
                if not success:
                    return False, f"位置设置失败: {message}"

            # 设置速度
            if speed is not None:
                success, message = self.set_gripper_speed(speed)
                results.append(("速度", success, message))
                if not success:
                    return False, f"速度设置失败: {message}"

            # 如果没有设置任何参数，直接返回
            if not results:
                return False, "未设置任何参数"
            
            # 非阻塞模式：立即返回
            if block == 0:
                success_count = sum(1 for _, success, _ in results if success)
                total_count = len(results)
                
                if success_count == total_count:
                    return True, f"所有参数设置指令已发送 ({success_count}/{total_count})"
                else:
                    failed_params = [name for name, success, _ in results if not success]
                    return False, f"部分参数设置失败: {', '.join(failed_params)}"
            
            # 阻塞模式：等待夹爪状态变为0（就绪状态）
            elif block == 1:
                logger.info("阻塞模式：等待夹爪状态变为就绪(status==1)...")
                start_time = time.time()
                last_status = None
                
                while time.time() - start_time < timeout:
                    # 获取当前夹爪状态
                    with self.gripper_data_lock:
                        current_status = self.current_gripper_status
                        current_position = self.current_gripper_position
                    print("当前状态: {}, 位置: {}".format(current_status, current_position))
                    # 记录状态变化
                    if current_status != last_status:
                        # logger.debug(f"夹爪状态: {current_status}, 位置: {current_position}")
                        last_status = current_status
                    
                    # 状态变为0时表示就绪，可以返回
                    # if current_status != 0 and last_status == 0:
                    if abs(current_position-position)<20:
                        success_count = sum(1 for _, success, _ in results if success)
                        total_count = len(results)
                        
                        if success_count == total_count:
                            return True, f"所有参数设置成功并已就绪 ({success_count}/{total_count})，等待时间: {time.time()-start_time:.2f}秒"
                        else:
                            failed_params = [name for name, success, _ in results if not success]
                            return False, f"参数设置部分失败: {', '.join(failed_params)}"
                    
             
                    # 短暂等待后继续检查
                    time.sleep(check_interval)
                
                # 超时处理
                return False, f"等待夹爪就绪超时({timeout}秒)，最终状态: {self.current_gripper_status}, 最终位置: {self.current_gripper_position}"
            
            else:
                return False, f"无效的阻塞模式参数: {block}，请使用0(非阻塞)或1(阻塞)"
                
        except Exception as e:
            logger.exception("设置夹爪参数过程异常")
            return False, f"设置过程异常: {str(e)}"
    # def set_gripper_params(self, position=None, force=None, speed=None):
    #     """
    #     综合设置夹爪参数
    #     Args:
    #         position: 位置 (0-1000 千分比)
    #         force: 力值 (20-100 百分比)
    #         speed: 速度 (1-100 百分比)
    #     """
    #     logger.info(f"设置夹爪参数 - 位置: {position}, 力: {force}, 速度: {speed}")
        
    #     results = []
        
    #     # 设置力值
    #     if force is not None:
    #         success, message = self.set_gripper_force(force)
    #         results.append(("力值", success, message))
    #         if not success:
    #             return False, f"力值设置失败: {message}"

    #     # 设置位置
    #     if position is not None:
    #         success, message = self.set_gripper_position(position)
    #         results.append(("位置", success, message))
    #         if not success:
    #             return False, f"位置设置失败: {message}"

    #     # 设置速度
    #     if speed is not None:
    #         success, message = self.set_gripper_speed(speed)
    #         results.append(("速度", success, message))
    #         if not success:
    #             return False, f"速度设置失败: {message}"

    #     # 汇总结果
    #     if results:
    #         success_count = sum(1 for _, success, _ in results if success)
    #         total_count = len(results)
            
    #         if success_count == total_count:
    #             return True, f"所有参数设置成功 ({success_count}/{total_count})"
    #         else:
    #             failed_params = [name for name, success, _ in results if not success]
    #             return False, f"部分参数设置失败: {', '.join(failed_params)}"
    #     else:
    #         return False, "未设置任何参数"

    def save_gripper_config(self):
        """保存夹爪配置到Flash"""
        logger.info("保存夹爪配置到Flash")
        
        try:
            command_frame = self.gripper_builder.build_save_command()
            return self._send_gripper_command(command_frame)
            
        except Exception as e:
            logger.exception("保存配置异常")
            return False, f"保存配置异常: {str(e)}"
    def _refresh_gripper_signals(self):
        """
        配置并刷新夹爪信号量[5](@ref)
        """
        if not self.is_connected or not self.robot:
            return False
            
        try:
            # 配置状态运行信号
            ret1 = self.robot.update_tio_rs_signal({
                'sig_name': self.gripper_signal_config['status_run'],
                'frequency': self.gripper_signal_frequency
            })
            
            # 配置位置信号
            ret2 = self.robot.update_tio_rs_signal({
                'sig_name': self.gripper_signal_config['position'], 
                'frequency': self.gripper_signal_frequency
            })
            
            if ret1[0] == 0 and ret2[0] == 0:
                logger.debug("夹爪信号量刷新配置成功")
                return True
            else:
                logger.warning("夹爪信号量配置失败: status_run={}, position={}".format(ret1[0], ret2[0]))
                return False
                
        except Exception as e:
            logger.error("刷新夹爪信号配置异常: {}".format(str(e)))
            return False

    def _update_gripper_status(self):
        """
        获取最新的夹爪状态和位置信息[5](@ref)
        返回: (success, status_run, position)
        """
        if not self.is_connected or not self.robot:
            return False, 0, 0
        
        try:
            # 获取所有RS485信号信息
            ret, sign_info_list = self.robot.get_rs485_signal_info()
            if ret != 0:
                return False, 0, 0
            
            status_run = 0
            position = 0
            found_status = False
            found_position = False
            
            # 遍历信号列表，查找目标信号
            for sig_info in sign_info_list:
                sig_name = sig_info.get('sig_name', '')
                sig_value = sig_info.get('value', 0)
                sig_addr = sig_info.get('sig_addr', 0)
                if sig_addr == self.gripper_signal_config['status_run']:
                    status_run = sig_value
                    found_status = True
                elif sig_addr == self.gripper_signal_config['position']:
                    position = sig_value
                    found_position = True
                
                # 如果两个信号都找到，提前退出
                if found_status and found_position:
                    break
            
            return True, status_run, position
            
        except Exception as e:
            logger.error("获取夹爪状态异常: {}".format(str(e)))
            return False, 0, 0


    def _gripper_monitor_loop(self):
        """
        夹爪状态监测线程主循环
        """
        logger.info("夹爪状态监测线程启动，频率: {}Hz".format(self.gripper_signal_frequency))
        
        # 初始配置信号量
        # if not self._refresh_gripper_signals():
        #     logger.error("夹爪信号量初始配置失败，监测线程退出")
        #     return
        
        while self.gripper_monitoring and self.is_connected:
            try:
                # 获取最新状态
                success, status_run, position = self._update_gripper_status()
                # print("夹爪状态: {}, 位置: {}".format(status_run, position))
                if success:
                    with self.gripper_data_lock:
                        old_status = self.current_gripper_status
                        self.current_gripper_status = status_run
                        self.current_gripper_position = position
                       
                    # 状态变化时触发回调
                    if old_status != status_run:
                        self._trigger_status_callbacks(status_run, position)
                        logger.debug("夹爪状态变化: {} -> {}, 位置: {}".format(
                            old_status, status_run, position))
                
                # 根据频率计算等待时间
                interval = 1.0 / self.gripper_signal_frequency
                time.sleep(interval)
                
            except Exception as e:
                logger.error("夹爪监测循环异常: {}".format(str(e)))
                time.sleep(0.1)  # 异常时等待100ms
        
        logger.info("夹爪状态监测线程退出")

    def _trigger_status_callbacks(self, status: int, position: int):
        """触发状态回调函数"""
        print("夹爪状态回调 - 状态: {}, 位置: {}".format(status, position))
        if self.gripper_status_callbacks is None:
            return
        for callback in self.gripper_status_callbacks:
            try:
                callback(status, position)
            except Exception as e:
                logger.error("夹爪状态回调执行异常: {}".format(str(e)))

    def start_gripper_monitoring(self, frequency: int = 1):
        """
        启动夹爪状态监测
        Args:
            frequency: 信号刷新频率(1-20 Hz)
        """
        if self.gripper_monitoring:
            logger.warning("夹爪监测已在运行中")
            return True
        
       
        
        self.gripper_signal_frequency = frequency
        self.gripper_monitoring = True
        
        self.gripper_monitor_thread = threading.Thread(
            target=self._gripper_monitor_loop, 
            daemon=True,
            name="GripperMonitor"
        )
        self.gripper_monitor_thread.start()
        
        logger.info("夹爪状态监测已启动，频率: {}Hz".format(frequency))
        return True

    def stop_gripper_monitoring(self):
        """停止夹爪状态监测"""
        self.gripper_monitoring = False
        if self.gripper_monitor_thread and self.gripper_monitor_thread.is_alive():
            self.gripper_monitor_thread.join(timeout=2.0)
        logger.info("夹爪状态监测已停止")

    def grasp(self, timeout: float = 10.0) -> tuple:
        """
        智能抓取：夹爪闭合过程中监测状态，遇到阻力(status_run=2)立即停止
        Args:
            timeout: 抓取超时时间(秒)
        Returns:
            tuple: (成功状态, 消息)
        """
        if not self.is_connected:
            return False, "机械臂未连接"
        
        logger.info("开始智能抓取过程，超时时间: {}秒".format(timeout))
        
        # 确保监测线程运行
        if not self.gripper_monitoring:
            self.start_gripper_monitoring()
            time.sleep(0.2)  # 等待监测线程稳定
        
        try:
            # 1. 设置夹爪开始闭合（位置为1）
            success, message = self.set_gripper_params(position=10,force=FORCE,speed=SPEED)
            if not success:
                return False, "夹爪闭合指令发送失败: {}".format(message)
            
            logger.info("夹爪开始闭合，监测状态变化...")
            start_time = time.time()
            last_position = 0
            stop_position = None
            last_status = 0
            # 2. 监测循环
            while time.time() - start_time < timeout:
                # 获取当前状态
                with self.gripper_data_lock:
                    current_status = self.current_gripper_status
                    current_position = self.current_gripper_position
                print("当前状态: {}, 位置: {}".format(current_status, current_position))
                # 记录位置变化
                if current_position != last_position:
                    logger.debug("夹爪位置: {}, 状态: {}".format(current_position, current_status))
                    last_position = current_position
                if current_status != last_status:
                        logger.debug(f"夹爪状态: {current_status}, 位置: {current_position}")
                        last_status = current_status
                    
                    # 状态变为0时表示就绪，可以返回
                   
                # 状态变为2时立即停止
                if current_status == 2:
                    stop_position = current_position
                    logger.info("检测到阻力(状态=2)，当前位置: {}，立即停止".format(stop_position))
                    
                    # 设置目标位置为当前位置
                    success, message = self.set_gripper_params(position=stop_position,force=FORCE,speed=SPEED)
                    if success:
                        return True, "抓取完成，夹爪停止在位置: {}".format(stop_position)
                    else:
                        return False, "抓取完成但停止指令发送失败: {}".format(message)
                
                # 检查是否已完全闭合
                if current_position <= 10:  # 允许误差
                    logger.info("夹爪已完全闭合，位置: {}".format(current_position))
                    return True, "夹爪已完全闭合"
                
                # 短暂等待后继续监测
                time.sleep(0.01)  # 10ms检查间隔
            
            # 超时处理
            return False, "抓取超时，最终位置: {}，状态: {}".format(
                self.current_gripper_position, self.current_gripper_status)
            
        except Exception as e:
            logger.exception("智能抓取过程异常")
            return False, "抓取过程异常: {}".format(str(e))

    # -------------------- 原有机械臂控制函数 --------------------
    def connect(self, heartbeat_interval: float = 0.02):
        logger.info("正在连接机械臂...")
        try:
            self.robot = jkrc.RC(self.ip)
            logger.info("机械臂实例创建成功")
                
            ret = self.robot.login()
            if ret[0] == 0:
                self.is_connected = True
                logger.info("机械臂连接成功")
                self.start_heartbeat()
                logger.info("机械臂心跳开始")
                return True, "连接成功"
            else:
                logger.error("机械臂登录失败，错误码: %s", ret[0])
                return False, f"登录失败，错误码: {ret[0]}"
        except Exception as e:
            logger.exception("连接异常")
            return False, f"连接异常: {str(e)}"

    def disconnect(self):
        logger.info("正在断开连接...")
        try:
            if self.robot:
                self.stop_heartbeat()
                self.robot.logout()
                self.is_connected = False
                logger.info("断开连接成功")
                return True, "断开成功"
        except Exception as e:
            logger.exception("断开异常")
            return False, f"断开异常: {str(e)}"

    def power_on(self):
        if not self.is_connected:
            logger.warning("上电失败：未连接")
            return False, "未连接"
        try:
            ret = self.robot.power_on()
            if ret[0] == 0:
                logger.info("上电成功")
                return True, "上电成功"
            else:
                logger.error("上电失败，错误码: %s", ret[0])
                return False, f"上电失败，错误码: {ret[0]}"
        except Exception as e:
            logger.exception("上电异常")
            return False, f"上电异常: {str(e)}"

    def enable(self):
        if not self.is_connected:
            logger.warning("使能失败：未连接")
            return False, "未连接"
        try:
            ret = self.robot.enable_robot()
            if ret[0] == 0:
                logger.info("使能成功")
                return True, "使能成功"
            else:
                logger.error("使能失败，错误码: %s", ret[0])
                return False, f"使能失败，错误码: {ret[0]}"
        except Exception as e:
            logger.exception("使能异常")
            return False, f"使能异常: {str(e)}"

    # -------------------- 下使能 --------------------
    def disable(self):
        if not self.is_connected:
            logger.warning("下使能失败：未连接")
            return False, "未连接"
        try:
            ret = self.robot.disable_robot()
            if ret[0] == 0:
                logger.info("下使能成功")
                return True, "下使能成功"
            else:
                logger.error("下使能失败，错误码: %s", ret[0])
                return False, f"下使能失败，错误码: {ret[0]}"
        except Exception as e:
            logger.exception("下使能异常")
            return False, f"下使能异常: {str(e)}"

    # -------------------- 获取当前位置 --------------------
    def get_current_position(self):
        if not self.is_connected:
            logger.warning("获取位置失败：未连接")
            return False, "未连接"
        try:
            ret_j = self.robot.get_joint_position()
            if ret_j[0] == 0:
                self.current_joints = list(ret_j[1])
            ret_c = self.robot.get_tcp_position()
            if ret_c[0] == 0:
                self.current_cartesian = list(ret_c[1])
            logger.info("关节: %s, 笛卡尔: %s", self.current_joints, self.current_cartesian)
            return True, "位置更新成功"
        except Exception as e:
            logger.exception("获取位置异常")
            return False, f"获取位置异常: {str(e)}"
    def get_current_TCP(self):
        if not self.is_connected:
            logger.warning("获取位置失败：未连接")
            return False, "未连接"
        try:
            # 返回值为（0，current_cartesian），current_cartesian结构为[x,y,z,rx,ry,rz]
            ret_c = self.robot.get_tcp_position()
            if ret_c[0] == 0:
                self.current_cartesian = list(ret_c[1])
            logger.info("TCP: %s", self.current_cartesian)
            return True, self.current_cartesian
        except Exception as e:
            logger.exception("获取位置异常")
            return False, f"获取位置异常: {str(e)}"
    # -------------------- 关节运动 --------------------
    def move_joints(self, target_joints, speed=None):
        if not self.is_connected:
            logger.warning("关节运动失败：未连接")
            return False, "未连接"
        try:
            speed = speed or self.joint_speed
            logger.info("关节运动开始，目标: %s, 速度: %s", target_joints, speed)
            ret = self.robot.joint_move(target_joints, 0, True, speed)
            if ret[0] == 0:
                logger.info("关节运动完成")
                return True, "关节运动完成"
            else:
                logger.error("关节运动失败，错误码: %s", ret[0])
                return False, f"关节运动失败，错误码: {ret[0]}"
        except Exception as e:
            logger.exception("关节运动异常")
            return False, f"关节运动异常: {str(e)}"

    # -------------------- 直线运动 --------------------
    def move_linear(self, target_cartesian, is_block=True, speed=None):
        if not self.is_connected:
            logger.warning("直线运动失败：未连接")
            return False, "未连接"
        try:
            speed = speed or self.linear_speed
            logger.info("直线运动开始，目标: %s, 速度: %s", target_cartesian, speed)
            ret = self.robot.linear_move(target_cartesian, 0, is_block, speed)
            if ret[0] == 0:
                logger.info("直线运动完成")
                return True, "直线运动完成"
            else:
                logger.error("直线运动失败，错误码: %s", ret[0])
                return False, f"直线运动失败，错误码: {ret[0]}"
        except Exception as e:
            logger.exception("直线运动异常")
            return False, f"直线运动异常: {str(e)}"
    def move_linear_extend(self, target_cartesian, _speed=None,_accel=None,TOL= 0.1,_is_block=True,_move_mode=0):
        if not self.is_connected:
            logger.warning("直线运动失败：未连接")
            return False, "未连接"
        try:
            if _speed is None:  _speed = self.linear_speed
            if _accel is None: _accel=self.linear_accel
            
            logger.info("直线运动开始，目标: %s, 速度: %s,加速度：%s", target_cartesian, _speed,_accel)
            ret = self.robot.linear_move_extend(end_pos=target_cartesian, move_mode=_move_mode, is_block=_is_block, speed=_speed,acc=_accel,tol=TOL)
            if ret[0] == 0:
                logger.info("直线运动完成")
                return True, "直线运动完成"
            else:
                logger.error("直线运动失败，错误码: %s", ret[0])
                return False, f"直线运动失败，错误码: {ret[0]}"
        except Exception as e:
            logger.exception("直线运动异常")
            return False, f"直线运动异常: {str(e)}"
    def move_circle(self, end_cartesian, mid_cartesian, _speed=None, _accel=None, TOL=0.1, _is_block=True, _move_mode=0):
        """
        执行圆弧运动
        :param end_cartesian: 结束点坐标 [x, y, z, rx, ry, rz]
        :param mid_cartesian: 中间点(路径点)坐标 [x, y, z, rx, ry, rz]
        :param _speed: 速度 (mm/s)，若为None则使用默认值
        :param _accel: 加速度 (mm/s^2)，若为None则使用默认值
        :param TOL: 终点误差
        :param _is_block: 是否阻塞
        :param _move_mode: 0-绝对, 1-增量, 2-连续
        :return: (bool, msg)
        """
        if not self.is_connected:
            logger.warning("圆弧运动失败：未连接")
            return False, "未连接"
        try:
            # 如果未指定速度和加速度，通常使用类中定义的线性运动默认值
            if _speed is None: _speed = self.linear_speed
            if _accel is None: _accel = self.linear_accel
            
            logger.info("圆弧运动开始，中间点: %s, 目标: %s, 速度: %s, 加速度: %s", 
                        mid_cartesian, end_cartesian, _speed, _accel)
            
            # 调用底层SDK的circular_move函数
            ret = self.robot.circular_move(
                end_pos=end_cartesian, 
                mid_pos=mid_cartesian, 
                move_mode=_move_mode, 
                is_block=_is_block, 
                speed=_speed, 
                acc=_accel, 
                tol=TOL
            )
            
            # 判断返回值，0为成功
            if ret[0] == 0:
                logger.info("圆弧运动完成")
                return True, "圆弧运动完成"
            else:
                logger.error("圆弧运动失败，错误码: %s", ret[0])
                return False, f"圆弧运动失败，错误码: {ret[0]}"
                
        except Exception as e:
            logger.exception("圆弧运动异常")
            return False, f"圆弧运动异常: {str(e)}"
    # -------------------- 停止运动 --------------------
    def stop_motion(self):
        if not self.is_connected:
            logger.warning("停止运动失败：未连接")
            return False, "未连接"
        try:
            ret = self.robot.jog_stop(-1)
            if ret[0] == 0:
                # logger.info("运动已停止")
                return True, "运动已停止"
            else:
                logger.error("停止运动失败，错误码: %s", ret[0])
                return False, "停止异常"
        except Exception as e:
            logger.exception("停止运动异常")
            return False, f"停止运动异常: {str(e)}"

    # -------------------- 设置工具坐标系 --------------------
    def set_tool_data(self, tool_length):
        if not self.is_connected:
            logger.warning("设置工具坐标系失败：未连接")
            return False, "未连接"
        try:
            tool_data = [0, 0, tool_length, 0, 0, 0]
            ret = self.robot.set_tool_data(1, tool_data, "custom_tool")
            if ret[0] == 0:
                ret = self.robot.set_tool_id(1)
                if ret[0] == 0:
                    logger.info("工具坐标系设置成功，长度: %s mm", tool_length)
                    return True, f"工具坐标系设置成功，长度: {tool_length}mm"
            logger.error("工具设置失败，错误码: %s", ret[0])
            return False, f"工具设置失败，错误码: {ret[0]}"
        except Exception as e:
            logger.exception("设置工具异常")
            return False, f"设置异常: {str(e)}"

    #将右臂坐标系下的坐标转为其他坐标系下的坐标
    def transform_right(matrix):
    # 这里的 matrix 是一个 4x4 的变换矩阵
    # 提取平移部分
        translation = matrix[:3, 3]
    # 提取旋转部分
        rotation = matrix[:3, :3]
        return {"translation": translation, "rotation": rotation}
    
    #其他坐标系下的坐标转为右臂坐标系下的坐标
    def reverse_transform_right(matrix, position):
        # 这里的 matrix 是一个 4x4 的变换矩阵，为原本将右臂坐标系转换到其他坐标系的矩阵
        # 这里的 position 是一个 3D 坐标点
        # 先将 position 转换为齐次坐标
        homogenous_position = np.append(position, 1)
        # 进行逆变换
        transformed = np.linalg.inv(matrix) @ homogenous_position
        # 返回转换后的 3D 坐标
        return transformed[:3]

    # -------------------- 点动 --------------------
    def jog(self, aj_num, coord_type, jog_vel):
        if not self.is_connected:
            logger.warning("点动失败：未连接")
            return False, "未连接"
        try:
            self.robot.jog(aj_num=aj_num, move_mode=2, coord_type=coord_type,
                           jog_vel=jog_vel, pos_cmd=0)
            logger.info("点动开始，轴/坐标类型: %s/%s, 速度: %s", aj_num, coord_type, jog_vel)
            return True, "点动开始"
        except Exception as e:
            logger.exception("点动异常")
            return False, f"点动错误: {str(e)}"

    # -------------------- 心跳 --------------------
    def start_heartbeat(self):
        if self.heartbeat_running:
            return
        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        
        self.start_gripper_monitoring()
        logger.info("心跳线程已启动")

    def stop_heartbeat(self):
        logger.info("心跳线程准备停止...")
        self.heartbeat_running = False

    def _heartbeat_loop(self):
        while self.heartbeat_running and self.is_connected:
            try:
                ret_j = self.robot.get_joint_position()
                if ret_j[0] == 0:
                    self.current_joints = list(ret_j[1])
                ret_c = self.robot.get_tcp_position()
                if ret_c[0] == 0:
                    self.current_cartesian = list(ret_c[1])
            except Exception:
                pass
            time.sleep(0.02)
        logger.info("心跳线程已退出")



# ...existing code...
ROBOT_IP = "192.168.1.25"
# 使用示例
if __name__ == "__main__":
    
    robot = JakaRobot(ROBOT_IP)
    
    #连接机械臂
    success, message = robot.connect()
    
    if success:
        #上电和使能
        robot.power_on()
        robot.enable()
        robot.init_gripper(0x01)
        robot.set_gripper_params(position=1000, force=30, speed=50,block=1)
        print("开始抓取")
        success, message = robot.grasp(timeout=8.0)
        if success:
            print("抓取成功: {}".format(message))
        else:
            print("抓取失败: {}".format(message))
        time.sleep(1)
        ret=robot.robot.get_rs485_signal_info()
        print(ret)
        # while True:
           
        #     robot.set_gripper_params(position=500, force=30, speed=50,block=1)
        #     time.sleep(0.1)
        robot.set_gripper_params(position=800, force=30, speed=50,block=1)
