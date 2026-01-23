# coding=utf-8

"""
眼在手外 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂基坐标系的 旋转矩阵和平移向量

"""

import os
import logging

import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from libs.auxiliary import find_latest_data_folder
from libs.log_setting import CommonLog

from save_poses2 import poses2_main

np.set_printoptions(precision=8,suppress=True)

logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)


current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"eye_hand_data")
logger_.info(f'current_path： {current_path}')
images_path = os.path.join("/home/zsl/lzk/project_11.19/Modules/hand_eye_calibration-main/eye_hand_data",find_latest_data_folder(current_path))
logger_.info(f'images_path :{images_path}')
file_path = os.path.join(images_path,"poses.txt")  #采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应


with open("/home/zsl/lzk/project_11.19/Modules/hand_eye_calibration-main/config.yaml", 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

XX = data.get("checkerboard_args").get("XX") #标定板的中长度对应的角点的个数
YY = data.get("checkerboard_args").get("YY") #标定板的中宽度对应的角点的个数
L = data.get("checkerboard_args").get("L")   #标定板一格的长度  单位为米

def func():

    path = os.path.dirname(__file__)
    print(path)

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)     # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = L*objp

    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点

    images_num = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
    detected_num =0
    for i in range(1, len(images_num) + 1):   #标定好的图片在images_path路径下，从0.jpg到x.jpg

        image_file = os.path.join(images_path,f"{i}.jpg")

        if os.path.exists(image_file):

            logger_.info(f'读 {image_file}')

            img = cv2.imread(image_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)

            if ret:
                detected_num += 1
                logger_.info(f"图像 {i} 检测到棋盘格角点")
                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

    N = len(img_points)
    logger_.info(f"总共{len(images_num)} 张图片中, {detected_num}张检测到棋盘格角点")
    # 标定,得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    logger_.info(f"内参矩阵:\n:{mtx}" ) # 内参数矩阵
    logger_.info(f"畸变系数:\n:{dist}")  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    logger_.info(f"标定板相对相机的旋转矩阵:\n:{rvecs}" ) # 内参数矩阵
    logger_.info(f"标定板相对相机的平移向量:\n:{tvecs}")  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    print("-----------------------------------------------------")
   # 在所有图像上标注标定板位姿
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.jpg')], 
                     key=lambda x: int(x.split('.')[0]))

    # 创建存放标注图片的文件夹
    annotated_folder = os.path.join(images_path, "annotated_images")
    os.makedirs(annotated_folder, exist_ok=True)

    for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        # 读取对应图像
        image_file = os.path.join(images_path, image_files[idx])
        img = cv2.imread(image_file)
        
        if img is not None:
            # 绘制坐标轴
            axis = np.float32([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]) * L * 5  # 坐标轴长度
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
            
            # 绘制棋盘格角点
            corners_img = img_points[idx]
            img_show = img.copy()
            cv2.drawChessboardCorners(img_show, (XX, YY), corners_img, True)
            
            # 绘制坐标轴
            corner = tuple(map(int, imgpts[0].ravel()))
            img_show = cv2.line(img_show, corner, tuple(map(int, imgpts[1].ravel())), (0,0,255), 5)  # X轴-蓝色
            img_show = cv2.line(img_show, corner, tuple(map(int, imgpts[2].ravel())), (0,255,0), 5)  # Y轴-绿色
            img_show = cv2.line(img_show, corner, tuple(map(int, imgpts[3].ravel())), (255,0,0), 5)  # Z轴-红色
            
            # 添加图像信息
            cv2.putText(img_show, f"Image {idx+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_show, f"X-Y-Z: B-G-R", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 添加平移向量信息
            tvec_text = f"T: [{tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f}]"
            cv2.putText(img_show, tvec_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # # 添加旋转矩阵信息（转换为欧拉角以便显示）
            rvec_text = f"Rvec: [{rvec[0][0]:.2f}, {rvec[1][0]:.2f}, {rvec[2][0]:.2f}]"
            cv2.putText(img_show, rvec_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # 保存带标注的图像到新文件夹
            output_path = os.path.join(annotated_folder, f"annotated_{idx+1}.jpg")
            cv2.imwrite(output_path, img_show)
            
            logger_.info(f"已保存标注图像: {output_path}")
    poses2_main(file_path)
    # 机器人末端在基座标系下的位姿

    csv_file = os.path.join(path,"RobotToolPose.csv")
    tool_pose = np.loadtxt(csv_file,delimiter=',')

    R_tool = []#末端在基座标系下的旋转矩阵
    t_tool = []# 末端在基座标系下的平移向量

    for i in range(int(N)):

        R_tool.append(tool_pose[0:3,4*i:4*i+3])
        t_tool.append(tool_pose[0:3,4*i+3])
    logger_.info(f"末端对基座的平移向量:\n:{t_tool}") 
    R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)

    return R,t
# def func():
#     path = os.path.dirname(__file__)
#     print(path)

#     # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
#     criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

#     # 获取标定板角点的位置
#     objp = np.zeros((XX * YY, 3), np.float32)
#     objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)     # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
#     objp = L*objp

#     obj_points = []     # 存储3D点
#     img_points = []     # 存储2D点
#     rvecs_solvePnP = []  # 存储通过solvePnP得到的旋转向量
#     tvecs_solvePnP = []  # 存储通过solvePnP得到的平移向量

#     images_num = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
#     detected_num = 0
    
#     # 先进行相机标定获取内参
#     for i in range(1, len(images_num) + 1):
#         image_file = os.path.join(images_path, f"{i}.jpg")
#         if os.path.exists(image_file):
#             logger_.info(f'读 {image_file}')
#             img = cv2.imread(image_file)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             size = gray.shape[::-1]
#             ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
            
#             if ret:
#                 detected_num += 1
#                 logger_.info(f"图像 {i} 检测到棋盘格角点")
#                 obj_points.append(objp)
#                 corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
#                 if corners2 is not None:
#                     img_points.append(corners2)
#                 else:
#                     img_points.append(corners)

#     # 进行相机标定获取内参矩阵和畸变系数
#     ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    
#     logger_.info(f"内参矩阵:\n{mtx}")
#     logger_.info(f"畸变系数:\n{dist}")

#     # 重新处理每张图像，使用solvePnP独立获取位姿
#     for i in range(1, len(images_num) + 1):
#         image_file = os.path.join(images_path, f"{i}.jpg")
#         if os.path.exists(image_file):
#             img = cv2.imread(image_file)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
            
#             if ret:
#                 # 使用亚像素角点
#                 corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                
#                 # 使用solvePnP独立计算每张图像的位姿[2](@ref)
#                 success, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
                
#                 if success:
#                     rvecs_solvePnP.append(rvec)
#                     tvecs_solvePnP.append(tvec)
#                     logger_.info(f"图像 {i} - 独立位姿计算成功")
#                 else:
#                     logger_.warning(f"图像 {i} - solvePnP失败，使用零位姿")
#                     rvecs_solvePnP.append(np.zeros((3, 1)))
#                     tvecs_solvePnP.append(np.zeros((3, 1)))

#     N = len(rvecs_solvePnP)
#     logger_.info(f"成功为 {N} 张图像计算独立位姿")

    # # 输出独立计算的位姿信息
    # for i, (rvec, tvec) in enumerate(zip(rvecs_solvePnP, tvecs_solvePnP)):
    #     logger_.info(f"图像 {i+1} - 旋转向量: {rvec.flatten()}")
    #     logger_.info(f"图像 {i+1} - 平移向量: {tvec.flatten()}")

    # # 在所有图像上标注标定板位姿（使用独立计算的位姿）
    # image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.jpg')], 
    #                  key=lambda x: int(x.split('.')[0]))

    # annotated_folder = os.path.join(images_path, "annotated_images")
    # os.makedirs(annotated_folder, exist_ok=True)

    # for idx, (rvec, tvec) in enumerate(zip(rvecs_solvePnP, tvecs_solvePnP)):
    #     image_file = os.path.join(images_path, image_files[idx])
    #     img = cv2.imread(image_file)
        
    #     if img is not None:
    #         # 绘制坐标轴
    #         axis = np.float32([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]) * L * 5
    #         imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
            
    #         # 绘制棋盘格角点
    #         corners_img = img_points[idx]
    #         img_show = img.copy()
    #         cv2.drawChessboardCorners(img_show, (XX, YY), corners_img, True)
            
    #         # 绘制坐标轴
    #         corner = tuple(map(int, imgpts[0].ravel()))
    #         img_show = cv2.line(img_show, corner, tuple(map(int, imgpts[1].ravel())), (0,0,255), 5)
    #         img_show = cv2.line(img_show, corner, tuple(map(int, imgpts[2].ravel())), (0,255,0), 5)
    #         img_show = cv2.line(img_show, corner, tuple(map(int, imgpts[3].ravel())), (255,0,0), 5)
            
    #         # 添加图像信息
    #         cv2.putText(img_show, f"Image {idx+1} (独立计算)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #         cv2.putText(img_show, f"X-Y-Z: B-G-R", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
    #         # 添加平移向量信息
    #         tvec_text = f"T: [{tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f}]"
    #         cv2.putText(img_show, tvec_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    #         rvec_text = f"Rvec: [{rvec[0][0]:.2f}, {rvec[1][0]:.2f}, {rvec[2][0]:.2f}]"
    #         cv2.putText(img_show, rvec_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    #         output_path = os.path.join(annotated_folder, f"annotated_{idx+1}.jpg")
    #         cv2.imwrite(output_path, img_show)
            
    #         logger_.info(f"已保存标注图像: {output_path}")

    # # 机器人末端在基座标系下的位姿
    # csv_file = os.path.join(path, "RobotToolPose.csv")
    # tool_pose = np.loadtxt(csv_file, delimiter=',')
    
    # R_tool = []  # 末端在基座标系下的旋转矩阵
    # t_tool = []  # 末端在基座标系下的平移向量

    # for i in range(N):
    #     R_tool.append(tool_pose[0:3, 4*i:4*i+3])
    #     t_tool.append(tool_pose[0:3, 4*i+3])
    
    # logger_.info(f"末端对基座的平移向量:\n{t_tool}")
    
    # # 使用独立计算的位姿进行手眼标定[1,2](@ref)
    # # R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs_solvePnP, tvecs_solvePnP, cv2.CALIB_HAND_EYE_TSAI)
    # R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs_solvePnP, tvecs_solvePnP, cv2.CALIB_HAND_EYE_DANIILIDIS)
    # return R, t
if __name__ == '__main__':

    # 旋转矩阵
    rotation_matrix, translation_vector = func()

    # 将旋转矩阵转换为四元数
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    x, y, z = translation_vector.flatten()

    logger_.info(f"旋转矩阵是:\n {            rotation_matrix}")

    logger_.info(f"平移向量是:\n {            translation_vector}")

    logger_.info(f"四元数是：\n {             quaternion}")

