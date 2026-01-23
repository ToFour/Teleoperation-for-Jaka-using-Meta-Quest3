"""

眼在手上 计算得是 相机相对于机械臂末端 齐次变换矩阵
计算 这个矩阵需要得是  标定板相对于相机得次变换矩阵 * 相机相对于机械臂末端得齐次变换矩阵 * 机械臂末端相对于基座得齐次变换矩阵

机械臂末端相对于基座得齐次变换矩阵（也就是机械臂位姿变换得齐次变换矩阵）

"""
import csv

import numpy as np


def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz@Ry@Rx

    return R


def pose_to_homogeneous_matrix(pose):

    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]

    return H


def save_matrices_to_csv(matrices, file_name):

    rows, cols = matrices[0].shape
    num_matrices = len(matrices)
    combined_matrix = np.zeros((rows, cols * num_matrices))

    for i, matrix in enumerate(matrices):
        combined_matrix[:, i * cols: (i + 1) * cols] = matrix

    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in combined_matrix:
            csv_writer.writerow(row)


def poses_main(filepath):

    # 打开文本文件
    with open(filepath, "r", encoding="utf-8") as f:
        # 读取文件中的所有行
        lines = f.readlines()
    # 定义一个空列表，用于存储结果

    # 遍历每一行数据
    lines = [float(i) for line in lines for i in line.split(',')]

    matrices = []
    for i in range(0,len(lines),6):
        matrices.append(pose_to_homogeneous_matrix(lines[i:i+6]))


    # 将齐次变换矩阵列表存储到 CSV 文件中
    save_matrices_to_csv(matrices, f'RobotToolPose.csv')

