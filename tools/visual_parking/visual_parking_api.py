'''
Author: tianwenshan tianwenshan@navinfo.com
Date: 2025-01-08 16:25:49
LastEditors: tianwenshan tianwenshan@navinfo.com
LastEditTime: 2025-01-24 17:56:36
FilePath: /ori_code/visual_parking_api.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os, sys
#import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
#import torch

def save_avi(visual_path, save_path):
    colors = np.array(
        [
            [0, 0, 0, 255],
            [255, 120, 50, 255],  # barrier              orangey
            [255, 192, 203, 255],  # bicycle              pink
            [255, 255, 0, 255],  # bus                  yellow
            [0, 150, 245, 255],  # car                  blue
            [0, 255, 255, 255],  # construction_vehicle cyan
            [200, 180, 0, 255],  # motorcycle           dark orange
            [255, 0, 0, 255],  # pedestrian           red
            [255, 240, 150, 255],  # traffic_cone         light yellow
            [135, 60, 0, 255],  # trailer              brown
            [160, 32, 240, 255],  # truck                purple
            [255, 0, 255, 255],  # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137, 255],
            [75, 0, 75, 255],  # sidewalk             dard purple
            [150, 240, 80, 255],  # terrain              light green
            [230, 230, 250, 255],  # manmade              white
            [0, 175, 0, 255],  # vegetation           green
            [0, 255, 127, 255],  # ego car              dark cyan
            [255, 99, 71, 255],
            [0, 191, 255, 255]
        ]
    ).astype(np.uint8)

    mlab.options.offscreen = True

    voxel_size = 0.15
    pc_range =  [-15, -15, -2.4, 15, 15, 0] #[-15, -15, -2, 15, 15, 0]


    # visual_path = sys.argv[1]
    # save_path = sys.argv[2]
    fov_voxels = np.load(visual_path) / voxel_size

    # fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    # fov_voxels[:, 0] += pc_range[0]
    # fov_voxels[:, 1] += pc_range[1]
    # fov_voxels[:, 2] += pc_range[2]
    # 如果后视相机，过滤掉前面的点
    # fov_voxels = fov_voxels[fov_voxels[:, 1] > 0]


    # 定义视角配置
    views = [
        # {
        #     'name': 'top_view',
        #     'azimuth': 180.0, 
        #     'elevation': 0.0, 
        #     'distance': 62,
        #     'focalpoint': [0.5, 0.5, 0.5],
        #     'filter': None,
        #     'figure_size': (1392, 1392)
        # },
        # {
        #     'name': '45_degree_view',
        #     'azimuth': 45.0, 
        #     'elevation': 45.0, 
        #     'distance': 60,
        #     'focalpoint': [0.5, 0.5, 0.5],
        #     'filter': None,
        #     'figure_size': (1392, 1392)
        # },
        {
            'name': 'front_view',
            'azimuth': 90.0, 
            'elevation': 45.0, 
            'distance': 30,
            'focalpoint': [0.5, 0.5, 0.5],
            'filter': None,
            'figure_size': (1392, 1392)
        },
        # {
        #     'name': 'rear_view',
        #     'azimuth': -90.0, 
        #     'elevation': 75.0, 
        #     'distance': 40,
        #     'focalpoint': [0.5, -2, 0.5],
        #     'filter': lambda y: y > 0,  # 过滤后方点
        #     'figure_size': (1056, 696)
        # }
    ]
    
    # 遍历并生成每个视角的图像
    for view in views:
        # 创建图形，使用视角特定的大小
        figure = mlab.figure(size=view['figure_size'], bgcolor=(1, 1, 1))
        
        # 应用数据过滤
        if view['filter']:
            filter_mask = view['filter'](fov_voxels[:, 1])
            filtered_voxels = fov_voxels[filter_mask]
        else:
            filtered_voxels = fov_voxels

        # 检查数据是否为空
        if len(filtered_voxels) == 0:
            print(f"Warning: No point cloud data for {view['name']}")
            mlab.close(figure)
            continue
        
        # 绘制点云
        # colors = color_by_distance(filtered_voxels)
        # opacity = opacity_by_distance(filtered_voxels)
        plt_plot_fov = mlab.points3d(
            filtered_voxels[:, 0],
            filtered_voxels[:, 1],
            filtered_voxels[:, 2],
            # color_by_distance(filtered_voxels),  # 使用归一化距离作为标量
            color=(0, 1, 0),
            mode="cube",
            scale_factor=voxel_size - 0.05*voxel_size,
            opacity=1.0,  # 使用透明度映射
            colormap='Greens'  # 使用对应的颜色映射
        )
        
        # 添加参考点
        mlab.points3d([0.5], [0.5], [-2.4+0.5], scale_factor=0.2, color=(1, 0, 0), mode='cube')
        mlab.points3d([0.5], [0.5-0.15], [-2.4+0.5], scale_factor=0.2, color=(1, 0, 0), mode='cube')
        mlab.points3d([0.5], [0.5+0.15], [-2.4+0.5], scale_factor=0.2, color=(1, 0, 0), mode='cube')  
        
        # 设置坐标轴和边框
        # mlab.outline(extent=[pc_range[0], pc_range[3], pc_range[1], pc_range[4], pc_range[2], pc_range[5]])
        # mlab.axes(extent=[pc_range[0], pc_range[3], pc_range[1], pc_range[4], pc_range[2], pc_range[5]], 
        #           nb_labels=5)
        
        # 设置视角
        mlab.view(
            azimuth=view['azimuth'], 
            elevation=view['elevation'], 
            distance=view['distance'], 
            focalpoint=view['focalpoint']
        )
        
        # 保存图像
        output_path = save_path.replace('.png', f'_{view["name"]}.png')
        mlab.savefig(output_path)
        print(f"Saved {view['name']} to {output_path}")
        
        # 关闭当前图形
        mlab.close(figure)

def color_by_distance(filtered_voxels, center=[0.5, 0.5, 0.5]):
    """根据距离生成颜色映射"""
    # 计算每个点到中心点的距离
    distances = np.sqrt(np.sum((filtered_voxels[:, :3] - center) ** 2, axis=1))
    
    # 归一化距离
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
    
    return 1.0 -normalized_distances

def opacity_by_distance(filtered_voxels, center=[0.5, 0.5, 0.5]):
    """根据距离生成透明度映射"""
    # 计算每个点到中心点的距离
    distances = np.sqrt(np.sum((filtered_voxels[:, :3] - center) ** 2, axis=1))
    
    # 归一化距离
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
    
    # 反转透明度，使近处更不透明
    opacity = 1.0 - normalized_distances
    
    # 添加指数衰减，使透明度变化更平滑
    opacity = np.power(opacity, 0.5)
    
    # 返回平均透明度
    return float(np.mean(opacity))
