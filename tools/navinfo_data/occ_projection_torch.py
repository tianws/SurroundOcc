#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R

@torch.jit.script
def fisheye_project_points_torch_jit(points_camera: torch.Tensor, 
                                     K: torch.Tensor, 
                                     D: torch.Tensor, 
                                     eps: float = 1e-8):
    """
    使用 TorchScript 优化的鱼眼相机投影函数
    
    :param points_camera: 相机坐标系点云 [N, 3]
    :param K: 相机内参矩阵 [3, 3]
    :param D: 畸变系数 [k1, k2, k3, k4]
    :param eps: 数值稳定性参数
    :return: 图像坐标
    """
    # 1. 计算归一化坐标
    x = points_camera[:, 0] / (points_camera[:, 2] + eps)
    y = points_camera[:, 1] / (points_camera[:, 2] + eps)
    
    # 2. 计算r²和r⁴
    r2 = x*x + y*y
    
    # 3. 计算 theta = atan(r)
    theta = torch.atan(torch.sqrt(r2 + eps))
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    
    # 4. 应用径向畸变
    theta_d = theta * (1 + D[0]*theta2 + D[1]*theta4 + D[2]*theta6 + D[3]*theta8)
    
    # 5. 计算缩放因子
    scale = torch.where(r2 > eps, 
                       theta_d / torch.sqrt(r2 + eps),
                       torch.ones_like(r2, device=x.device))
    
    # 6. 应用缩放得到畸变后的归一化坐标
    xd = x * scale
    yd = y * scale
    
    # 7. 应用相机内参矩阵
    u = K[0, 0] * xd + K[0, 2]  # fx * xd + cx
    v = K[1, 1] * yd + K[1, 2]  # fy * yd + cy
    
    return torch.stack([u, v], dim=1)

def fisheye_project_points_torch(points_camera, K, D, eps=1e-8):
    """
    使用 PyTorch 实现鱼眼相机投影，匹配 OpenCV fisheye.projectPoints 的实现
    
    :param points_camera: 相机坐标系点云 [N, 3]
    :param K: 相机内参矩阵 [3, 3]
    :param D: 畸变系数 [k1, k2, k3, k4]
    :param eps: 数值稳定性参数
    :return: 图像坐标
    """
    # 确保输入是 PyTorch 张量并移到 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not isinstance(points_camera, torch.Tensor):
        points_camera = torch.from_numpy(points_camera).float()
    if not isinstance(K, torch.Tensor):
        K = torch.from_numpy(K).float()
    if not isinstance(D, torch.Tensor):
        D = torch.from_numpy(D).float()
    
    points_camera = points_camera.to(device)
    K = K.to(device)
    D = D.to(device)

    # 使用 TorchScript 优化版本
    start_time = time.time()
    image_coords = fisheye_project_points_torch_jit(points_camera, K, D, eps)
    end_time = time.time()
    
    # 打印性能信息
    print(f"TorchScript 投影耗时: {(end_time - start_time) * 1000:.2f} ms")
    
    return image_coords.cpu().numpy()

def project_occ_to_fisheye_torch(occ_data, K, D, T_l2c, debug=False):
    """
    使用 PyTorch 将主激光雷达坐标系的点云坐标投影到鱼眼相机图像
    
    :param occ_data: 主激光雷达坐标系下的点云坐标，shape为(n, 3)
    :param K: 相机内参矩阵
    :param D: 鱼眼相机畸变系数
    :param T_l2c: 激光雷达到相机的变换矩阵
    :param debug: 是否开启调试模式
    :return: 投影后的图像坐标和相机坐标系点
    """
    # 确保输入是 PyTorch 张量
    if not isinstance(occ_data, torch.Tensor):
        points_lidar = torch.from_numpy(occ_data).float()
    else:
        points_lidar = occ_data.float()
    
    if not isinstance(K, torch.Tensor):
        K = torch.from_numpy(K).float()
    
    if not isinstance(D, torch.Tensor):
        D = torch.from_numpy(D).float()
    
    if not isinstance(T_l2c, torch.Tensor):
        T_l2c = torch.from_numpy(T_l2c).float()
    
    # 移动到 GPU
    points_lidar = points_lidar.cuda()
    K = K.cuda()
    D = D.cuda()
    T_l2c = T_l2c.cuda()
    
    # 添加齐次坐标
    points_lidar_homo = torch.cat([
        points_lidar, 
        torch.ones((points_lidar.shape[0], 1), device=points_lidar.device)
    ], dim=1)
    
    # 使用逆变换矩阵转换到相机坐标系
    T_l2c_inv = torch.inverse(T_l2c)
    points_camera_homo = torch.mm(T_l2c_inv, points_lidar_homo.t()).t()
    points_camera = points_camera_homo[:, :3]
    
    # 过滤正深度点
    mask = points_camera[:, 2] > 0
    points_camera_filtered = points_camera[mask]
    
    if debug:
        print(f"总输入点数: {points_lidar.shape[0]}")
        print(f"正深度点数: {torch.sum(mask).item()}")
        print(f"深度范围: [{points_camera[:, 2].min()}, {points_camera[:, 2].max()}]")
    
    # 使用 PyTorch 实现鱼眼投影
    image_coords = fisheye_project_points_torch(points_camera_filtered.cpu().numpy(), K.cpu().numpy(), D.cpu().numpy())
    
    return image_coords, points_camera_filtered.cpu().numpy()

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

def load_sensor_info(info_path):
    with open(info_path, 'r') as f:
        sensor_info = json.load(f)
    return sensor_info

def get_camera_params(sensor_info, cam_name):
    for camera in sensor_info['sensors']['Cameras']:
        if camera['name'] == cam_name:
            KD = camera['intrinsic']
            extrinsic = camera['extrinsic']
            return KD, extrinsic
    return None, None

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

    # 确保保存目录存在
    os.makedirs('visualizations_torch', exist_ok=True)

    for gts_file in gts_files[:5]:  # 处理前5个文件
        for cam_name in fish_eye_cams:
            print(f"处理文件: {gts_file}, 相机: {cam_name}")
            
            # 加载栅格数据
            try:
                occ_data = np.load(os.path.join(gts_dir, gts_file))
            except Exception as e:
                print(f"加载 {gts_file} 失败: {e}")
                continue

            # 获取相机参数
            try:
                KD, extrinsic = get_camera_params(sensor_info, cam_name)
                K = np.array(KD['K'])
                D = np.array(KD['D'])  # 假设畸变系数在外参中
            except Exception as e:
                print(f"获取 {cam_name} 参数失败: {e}")
                continue

            # 构建变换矩阵
            try:
                T_l2c = np.zeros((4,4))

                # 平移部分
                T_l2c[:3, 3] = extrinsic['to_lidar_main'][:3]

                # 四元数转旋转矩阵
                quat = extrinsic['to_lidar_main'][3:]
                rotation_matrix = R.from_quat(quat).as_matrix()
                T_l2c[:3, :3] = rotation_matrix

                T_l2c[3, 3] = 1
            except Exception as e:
                print(f"构建变换矩阵失败: {e}")
                continue

            # 查找对应的图像
            image_path = os.path.join(camera_dir, cam_name, gts_file.replace('.npy', '.jpg'))

            if os.path.exists(image_path):
                try:
                    # 投影并可视化
                    image_coords, points_camera = project_occ_to_fisheye_torch(
                        torch.from_numpy(occ_data).float().cuda(), 
                        torch.from_numpy(K).float().cuda(), 
                        torch.from_numpy(D).float().cuda(), 
                        torch.from_numpy(T_l2c).float().cuda(), 
                        debug=True
                    )
                    
                    # 保存可视化结果
                    save_path = os.path.join('visualizations_torch', f'{cam_name}_{gts_file.replace(".npy", ".png")}')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    visualize_projection(image_path, image_coords, points_camera, save_path=save_path)
                    print(f"已保存可视化结果到: {save_path}")
                
                except Exception as e:
                    print(f"投影或可视化失败: {e}")
                    continue
            else:
                print(f"图像未找到: {image_path}")


if __name__ == "__main__":
    main()
