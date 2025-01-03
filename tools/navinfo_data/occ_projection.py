#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def load_sensor_info(info_path):
    """
    加载传感器配置信息
    """
    with open(info_path, 'r') as f:
        sensor_info = json.load(f)
    return sensor_info


def get_camera_params(sensor_info, camera_name):
    """
    获取指定相机的内参和外参
    """
    for camera in sensor_info['sensors']['Cameras']:
        if camera['name'] == camera_name:
            return camera['intrinsic'], camera['extrinsic']
    raise ValueError(f"Camera {camera_name} not found in sensor info")


def project_occ_to_fisheye(occ_data, K, D, T_l2c):
    """
    将主激光雷达坐标系的点云坐标投影到鱼眼相机图像
    
    :param occ_data: 主激光雷达坐标系下的点云坐标，shape为(n, 3)
    :param K: 相机内参矩阵
    :param D: 鱼眼相机畸变系数
    :param T_l2c: 激光雷达到相机的变换矩阵
    :return: 投影后的图像坐标和相机坐标系点
    """

    points_lidar = occ_data

    points_lidar = np.hstack([points_lidar, np.ones((points_lidar.shape[0], 1))])
    points_camera = np.dot(np.linalg.inv(T_l2c), points_lidar.T)
    points_camera = points_camera[0:3, :].T

    # points -= T_l2c[:3, 3]
    # points = np.matmul(points)
    # points = np.matmul(points, T_l2c[:3, :3]) + T_l2c[:3, 3]

    mask = points_camera[:, 2] > 0
    points_camera = points_camera[mask]

    # projects with opencv project
    image_coords = cv2.fisheye.projectPoints(points_camera.reshape(1, -1, 3),
                                             np.zeros(3), np.zeros(3), K,
                                             D)[0].reshape(-1, 2)
    # image_coords = cv2.fisheye.distortPoints(points, K, D)[0].reshape(-1, 2)
    return image_coords, points_camera

def visualize_projection(image_path, image_coords, points_camera, save_path=None):
    """
    在鱼眼图像上可视化投影点并可选保存结果
    
    :param image_path: 原始图像路径
    :param image_coords: 投影坐标
    :param points_camera: 相机坐标系点
    :param save_path: 可选的保存路径，如果提供则保存可视化结果
    """
    img = cv2.imread(image_path)

    # 过滤在图像范围内的点
    mask = (image_coords[:, 0] >= 0) & (image_coords[:, 0] < img.shape[1]) & \
           (image_coords[:, 1] >= 0) & (image_coords[:, 1] < img.shape[0])

    valid_coords = image_coords[mask]
    valid_points = points_camera[mask]

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 根据点的深度着色
    depths = valid_points[:, 2]
    plt.scatter(valid_coords[:, 0], valid_coords[:, 1], c=depths, cmap='viridis',
                alpha=0.5, s=10, edgecolors='none')

    plt.colorbar(label='Depth (m)')
    plt.title(f'OCC Projection to {os.path.basename(image_path)}')
    plt.tight_layout()
    
    # 如果提供了保存路径，则保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()


def main():
    # 数据集路径
    dataset_path = '/home/datasets/ads_largedata/apa_parking_data/baidu_occ_data/datasets/2024-12-02-14-09-56'

    # 加载传感器信息
    info_path = os.path.join(dataset_path, 'info.json')
    sensor_info = load_sensor_info(info_path)

    # 选择一个gts文件和对应的鱼眼相机
    gts_dir = os.path.join(dataset_path, 'gts')
    camera_dir = os.path.join(dataset_path, 'Camera')

    # 选择第一个gts文件和第一个鱼眼相机
    gts_files = sorted(os.listdir(gts_dir))
    fish_eye_cams = ['fisheye_front', 'fisheye_right', 'fisheye_left', 'fisheye_back']

    for gts_file in gts_files[:5]:  # 处理前5个文件
        for cam_name in fish_eye_cams:
            # 加载栅格数据
            occ_data = np.load(os.path.join(gts_dir, gts_file))

            # 获取相机参数
            KD, extrinsic = get_camera_params(sensor_info, cam_name)
            K = np.array(KD['K'])
            D = np.array(KD['D'])  # 假设畸变系数在外参中

            # 构建变换矩阵
            T_l2c = np.zeros((4,4))

            # 平移部分
            T_l2c[:3, 3] = extrinsic['to_lidar_main'][:3]

            # 四元数转旋转矩阵
            quat = extrinsic['to_lidar_main'][3:]
            rotation_matrix = R.from_quat(quat).as_matrix()
            T_l2c[:3, :3] = rotation_matrix

            T_l2c[3, 3] = 1

            # 查找对应的图像
            image_path = os.path.join(camera_dir, cam_name, gts_file.replace('.npy', '.jpg'))

            if os.path.exists(image_path):
                # 投影并可视化
                image_coords, points_camera = project_occ_to_fisheye(occ_data, K, D, T_l2c)
                visualize_projection(image_path, image_coords, points_camera, save_path=os.path.join('visualizations', cam_name, gts_file.replace('.npy', '.png')))
            else:
                print(f"Image not found: {image_path}")


if __name__ == "__main__":
    main()
