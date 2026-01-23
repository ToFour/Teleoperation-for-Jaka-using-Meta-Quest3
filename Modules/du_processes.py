#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
from multiprocessing import shared_memory, Barrier
import time
import struct
import logging
import sys
import math
from jaka_control import JakaRobot  # 导入你提供的库

# -------------------- 配置参数 --------------------
LEFT_ARM_IP = "192.168.1.24"   # 左臂IP
RIGHT_ARM_IP = "192.168.1.25"  # 右臂IP
LOG_FREQUENCY = 5             # 数据记录频率 (Hz) - 建议改回10或20，1太慢了
RECORD_SECONDS = 10            # 预计运行/记录时长 (秒)

# 共享内存配置
# 数据格式: Sequence(Q), Timestamp(d), ArmID(i), Elapsed(d), Joints[6](6d), TCP[6](6d),grippos(d),gripforce(d),gripspeed(d)
STRUCT_FMT = 'Q d i d d d d d d d d d d d d d d d d'
RECORD_SIZE = struct.calcsize(STRUCT_FMT)
# 预留缓冲空间 (频率 * 秒数 * 2倍余量)
MAX_RECORDS = int(LOG_FREQUENCY * RECORD_SECONDS * 2) 

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="[%(processName)s] %(message)s", # 简化日志格式
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DualArmController")


class SharedMemoryBuffer:
    """
    共享内存管理器
    负责创建共享内存块，并提供写入和读取解析的方法
    """
    def __init__(self, name, create=False, size=0):
        self.name = name
        self.shm = None
        
        # 1. 创建或连接共享内存
        if create:
            # 主进程创建逻辑
            try:
                # 尝试创建
                self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
                # 初始化头部计数器为0
                self.shm.buf[:4] = struct.pack('I', 0)
            except FileExistsError:
                # 如果已存在，先销毁旧的再创建新的
                tmp = shared_memory.SharedMemory(name=name)
                tmp.unlink()
                self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
                self.shm.buf[:4] = struct.pack('I', 0)
        else:
            # 子进程连接逻辑
            try:
                self.shm = shared_memory.SharedMemory(name=name)
            except FileNotFoundError:
                raise Exception(f"共享内存 {name} 未找到，请确保主进程已创建它。")

        # 2. 【关键修复】动态获取真实的内存大小
        # 不要依赖传入的 num_records，直接读取底层对象的 size
        self.size = self.shm.size
        # logging.info(f"共享内存 '{name}' 初始化成功. 大小: {self.size} bytes")

    def write_record(self, index, data_tuple):
        """
        写入一条记录
        """
        try:
            offset = 4 + (index * RECORD_SIZE)  # 跳过头部4字节
            
            # 检查越界
            if offset + RECORD_SIZE > self.size:
                logger.error(f"内存溢出! Index {index} 超出范围 (Size: {self.size})")
                return False

            # 数据打包
            try:
                packed_data = struct.pack(STRUCT_FMT, *data_tuple)
            except struct.error as e:
                logger.error(f"数据打包失败: {e}. 数据: {data_tuple}")
                return False

            # 写入 buffer
            self.shm.buf[offset : offset + RECORD_SIZE] = packed_data

            # 更新头部计数器 (原子性无法完全保证，但在此场景下通常够用)
            self.shm.buf[:4] = struct.pack('I', index + 1)
            return True
            
        except Exception as e:
            logger.error(f"写入共享内存未知异常: {e}")
            return False

    def read_all_records(self):
        """读取所有记录并解析"""
        if not self.shm:
            return []
            
        try:
            count_bytes = bytes(self.shm.buf[:4])
            total_records = struct.unpack('I', count_bytes)[0]
            
            # 安全检查：如果记录数明显异常（可能是脏数据），限制读取
            max_possible = (self.size - 4) // RECORD_SIZE
            if total_records > max_possible:
                logger.warning(f"读取异常: 记录数 {total_records} 超过最大可能值 {max_possible}，可能内存未初始化")
                total_records = max_possible

            records = []
            for i in range(total_records):
                offset = 4 + (i * RECORD_SIZE)
                chunk = bytes(self.shm.buf[offset : offset + RECORD_SIZE])
                record = struct.unpack(STRUCT_FMT, chunk)
                records.append(record)
            return records
        except Exception as e:
            logger.error(f"读取数据失败: {e}")
            return []

    def close(self):
        """关闭连接"""
        if self.shm:
            self.shm.close()

    def unlink(self):
        """销毁共享内存"""
        if self.shm:
            try:
                self.shm.unlink()
            except Exception:
                pass

# 复位机械臂位姿与夹爪位姿
def robot_reset(self,speed):
    if self.gripper_name == "right_gripper":
        target_joint_re_right = [0,d2r(60),d2r(120),d2r(30),d2r(270),d2r(140)]
        JakaRobot.move_joints(self,target_joint_re_right,speed = speed)

    if self.gripper_name == "left_gripper":
        target_joint_re_left = [d2r(-180),d2r(120),d2r(-120),d2r(150),d2r(90),d2r(-220)]
        JakaRobot.move_joints(self,target_joint_re_left,speed = speed)
    
    JakaRobot.set_gripper_position(self,1000)
    logger.info("[%s] 机械臂与夹爪复位完成", self.gripper_name)


def robot_worker(arm_name, ip, shm_name, barrier, target_pose, arm_id):
    """
    机械臂工作进程
    """
    process_logger = logging.getLogger(f"Worker-{arm_name}")
    process_logger.info(f"进程启动,IP: {ip}")

    # 1. 连接共享内存 (注意：这里不需要 size 参数，因为它会自动获取)
    try:
        buffer = SharedMemoryBuffer(shm_name, create=False)
    except Exception as e:
        process_logger.error(f"无法连接共享内存: {e}")
        return

    # 2. 初始化机械臂
    robot = JakaRobot(ip)
    
    # 增加连接重试机制或明确报错
    try:
        success, msg = robot.connect()
        if not success:
            process_logger.error(f"连接机械臂失败: {msg}")
            return
    except Exception as e:
        process_logger.error(f"连接过程发生异常: {e}")
        return

    try:
        # 上电与使能
        robot.power_on()
        robot.enable()

        # 初始位置更新 (确保数据不为空)
        robot.get_current_position()
        process_logger.info(f"初始位置获取成功: TCP={robot.current_cartesian[:3]}")

        # 3. 同步等待
        process_logger.info("等待双臂同步...")
        try:
            barrier.wait(timeout=15) # 增加一点超时时间
        except Exception:
            process_logger.warning("同步等待超时! 另一只手臂可能未连接。将继续单臂执行。")
        
        process_logger.info("开始执行任务")

        # -------------------- 记录线程配置 --------------------
        import threading
        stop_event = threading.Event()
        
        def logging_task():
            """子线程：高频记录数据"""
            start_time = time.time()
            seq = 0
            process_logger.info("记录线程启动")

            while not stop_event.is_set():
                loop_start = time.time()
                try:
                    # 获取最新数据 (如果 JakaRobot 内部没有自动刷新线程，必须手动调 get)
                    robot.get_current_position() 
                    robot._update_gripper_status()

                    now = time.time()
                    elapsed = now - start_time
                    
                    # 获取数据副本，防止线程竞争导致的不一致
                    joints = list(robot.current_joints) if robot.current_joints else [0.0]*6
                    tcp = list(robot.current_cartesian) if robot.current_cartesian else [0.0]*6
                    g_pos = float(getattr(robot, 'current_gripper_position', 0))
                    g_force = float(getattr(robot, 'current_gripper_force', 50))
                    g_speed = float(getattr(robot, 'current_gripper_speed', 50))

                    # 补全数据维度 (防止None或长度不足导致 struct 报错)
                    if len(joints) < 6: joints += [0.0] * (6 - len(joints))
                    if len(tcp) < 6: tcp += [0.0] * (6 - len(tcp))

                    # 组装数据 [seq, time, id, elapsed, j1..j6, tcp..rz]
                    row_data = [seq, now, arm_id, elapsed] + joints[:6] + tcp[:6] + [g_pos, g_force, g_speed]
                    
                    # 写入
                    if buffer.write_record(seq, row_data):
                        # 【日志】每条都打印（调试用），生产环境建议改为 seq % 10 == 0
                        process_logger.info(f"Log OK: Seq={seq}, rd={row_data}")
                    else:
                        process_logger.error("写入Buffer失败 (可能已满)")

                    seq += 1

                except Exception as e:
                    process_logger.error(f"记录循环异常: {e}")

                # 频率控制
                rest = (1.0 / LOG_FREQUENCY) - (time.time() - loop_start)
                if rest > 0:
                    time.sleep(rest)

            process_logger.info(f"记录线程结束，共 {seq} 条")

        # 启动记录线程
        t_log = threading.Thread(target=logging_task, daemon=True)
        t_log.start()

        # -------------------- 运动逻辑 --------------------
        try:
            process_logger.info(f"开始移动到目标: {target_pose}")

            # 运动指令
            # 注意：speed单位通常是 mm/s 或 deg/s，根据你的库定义调整
            robot.move_linear(target_pose, is_block = False, speed=200) 
            
            # 模拟逻辑：左手比右手多停留一会
            if arm_name == "Left":
                time.sleep(3.0)
                process_logger.info("左手额外任务执行中...")
            
            # 简单的回程或者等待，确保有足够时间记录数据
            time.sleep(2.0)
            process_logger.info("运动逻辑完成")

        except Exception as e:
            process_logger.error(f"运动控制异常: {e}")
            
    except Exception as main_e:
        process_logger.error(f"主逻辑异常: {main_e}")
        
    finally:
        # 清理工作
        stop_event.set()
        if 't_log' in locals():
            t_log.join(timeout=2)
        
        # 保护性下电
        try:
            robot.disconnect()
        except:
            pass
            
        buffer.close()
        process_logger.info("进程退出")


def d2r(degree):
    """角度转弧度"""
    return degree * math.pi / 180

def main():
    logger.info("=== 双臂协同控制系统 V2.1 (Fix) ===")

    # 定义共享内存名称
    shm_name_left = "shm_jaka_left"
    shm_name_right = "shm_jaka_right"
    
    # 计算需要分配的总字节数
    # RECORD_SIZE * MAX_RECORDS + 4字节头部
    total_buffer_size = RECORD_SIZE * MAX_RECORDS + 4 

    # 1. 创建共享内存块
    try:
        shm_left = SharedMemoryBuffer(shm_name_left, create=True, size=total_buffer_size)
        shm_right = SharedMemoryBuffer(shm_name_right, create=True, size=total_buffer_size)
        logger.info(f"共享内存已创建. 容量: {MAX_RECORDS} 条记录 ({total_buffer_size} bytes)")
    except Exception as e:
        logger.error(f"内存创建失败: {e}")
        return

    # 2. 定义目标点
    # 注意：linear运动如果目标点不可达会报错，请确保点位安全
    target_left = [430.0, 140.0, 170.0, d2r(171.0), d2r(4.0), d2r(133.5)] 
    target_right = [480.0, -230.0, 183.0, d2r(175.0), d2r(4.0), d2r(121.0)]

    # 3. 创建同步屏障
    barrier = Barrier(2)

    # 4. 启动进程
    p_left = multiprocessing.Process(
        target=robot_worker,
        args=("Left", LEFT_ARM_IP, shm_name_left, barrier, target_left, 0),
        name="Proc-Left"
    )
    p_right = multiprocessing.Process(
        target=robot_worker,
        args=("Right", RIGHT_ARM_IP, shm_name_right, barrier, target_right, 1),
        name="Proc-Right"
    )

    p_left.start()
    p_right.start()

    logging.info("监控运行状态...")
    
    left_alive = True
    right_alive = True
    
    # 5. 循环监控，支持独立退出
    while left_alive or right_alive:
        # 检查左臂
        if left_alive and not p_left.is_alive():
            logging.info(">>> 左臂任务已结束")
            left_alive = False
            
        # 检查右臂
        if right_alive and not p_right.is_alive():
            logging.info(">>> 右臂任务已结束")
            right_alive = False
            
        time.sleep(0.5)

    logging.info("所有任务结束，导出数据...")

    # 6. 读取数据
    try:
        records_left = shm_left.read_all_records()
        records_right = shm_right.read_all_records()

        logger.info(f"左臂记录数: {len(records_left)}")
        logger.info(f"右臂记录数: {len(records_right)}")

        # 验证数据
        if records_left:
            # last = records_left[-1]
            # print(f"L最后一条: Seq={last[0]}, T={last[3]:.2f}s, TCP={last[10:13]}")
            for r in records_left[-5:]:
                print(f"L: {r}")
        
        
        if records_right:
            # last = records_right[-1]
            # print(f"R最后一条: Seq={last[0]}, T={last[3]:.2f}s, TCP={last[10:13]}")
            for r in records_right[-5:]:
                print(f"R: {r}")

    finally:
        # 7. 清理
        logger.info("清理共享内存...")
        shm_left.unlink()
        shm_right.unlink()
        logger.info("Done.")

if __name__ == "__main__":
    main()