# coding=utf-8
import json
import logging,os
import socket
import time
import sys
import numpy as np
import cv2
import pyrealsense2 as rs

from libs.log_setting import CommonLog
from libs.auxiliary import create_folder_with_date, get_ip, popup_message
sys.path.append("/home/zsl/lzk/project_11.19/Modules")
from jaka_control import*




cam0_origin_path = create_folder_with_date() # 提前建立好的存储照片文件的目录


logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

def callback(frame):

    scaling_factor = 1
    global count

    cv_img = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    cv2.imshow("Capture_Video", cv_img)  # 窗口显示，显示名为 Capture_Video

    k = cv2.waitKey(30) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
    # 按下 ‘s’ 键时，保存图片
    
    if k == ord('s'):  # 若检测到按键 ‘s’，打印字符串

        state,pose = robot.get_current_TCP()       # 发送命令并获取状态和位姿      
        pose[:3] = [p / 1000.0 for p in pose[:3]]  # 位置部分从毫米转换为米
        logger_.info(f'获取状态：{"成功" if state else "失败"}，{f"当前位姿为{pose}" if state else None}')
        if state:

            filename = os.path.join(cam0_origin_path,"poses.txt")

            with open(filename, 'a+') as f:
                # 将列表中的元素用空格连接成一行
                pose_ = [str(i) for i in pose]
                new_line = f'{",".join(pose_)}\n'
                # 将新行附加到文件的末尾
                f.write(new_line)

            image_path = os.path.join(cam0_origin_path,f"{str(count)}.jpg")
            cv2.imwrite(image_path , cv_img)
            logger_.info(f"===采集第{count}次数据！")

        count += 1

    else:
        pass



def DisplayandCalibration():

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280 , 720, rs.format.bgr8, 30) # 相机配置

    try:
        pipeline.start(config)
    except Exception as e:
        logger_.error_(f"相机连接异常：{e}")
        popup_message("提醒", "相机连接异常")

        sys.exit(1)

    global count
    count = 1

    logger_.info(f"开始手眼标定程序，当前程序版号V1.0.0")

    try:
        while True:
            frames = pipeline.wait_for_frames() #   等待一帧数据
            color_frame = frames.get_color_frame()  #   获取颜色帧对象
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())# 将帧对象转换成原始字节数据再转换为Numpy数组
            callback(color_image)

    finally:

        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    #jaka初始化
    robot = JakaRobot(ROBOT_IP)
    robot.connect()
    robot.power_on()
    robot.enable()
    robot.robot.set_tool_id(1)
    robot.set_tool_data(140) # 设置工具坐标系,确保TCP位姿获取正确
    
    #   标定循环

    DisplayandCalibration()
