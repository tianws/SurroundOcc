import os
import os.path as osp
import json
import glob
import numpy as np
import mmcv
from scipy.spatial.transform import Rotation as R
import sys


def create_baidu_occ_infos(root_paths, out_path, info_prefix='baidu_occ', val_folder=None):
    """
    从多个百度 OCC 数据集文件夹生成信息文件，模仿 NuScenes 数据格式
    
    Args:
        root_paths (list): 数据集根目录列表
        out_path (str): 输出目录
        info_prefix (str): 信息文件前缀
        val_folder (str, optional): 指定作为验证集的文件夹名。如果为 None，则随机选择一个
    """
    # 确保输出目录存在
    os.makedirs(out_path, exist_ok=True)
    
    # 定义需要的鱼眼相机列表
    fisheye_cameras = ['fisheye_front', 'fisheye_left', 'fisheye_right', 'fisheye_back']
    
    # 存储所有数据集的信息
    all_infos = []
    
    # 遍历所有数据集文件夹
    for root_path in root_paths:
        print(f"处理数据集文件夹：{root_path}")
        
        # 检查 info.json 是否存在
        info_json_path = os.path.join(root_path, 'info.json')
        if not os.path.exists(info_json_path):
            raise FileNotFoundError(f"错误：未找到 {info_json_path} 文件")
        
        # 加载传感器信息
        try:
            with open(info_json_path, 'r') as f:
                sensor_info = json.load(f)
            print(f"成功加载 {info_json_path}")
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(f"错误：解析 {info_json_path} 文件失败：{e}")
        
        # 检查 gts 目录是否存在
        gts_path = os.path.join(root_path, 'gts')
        if not os.path.exists(gts_path):
            raise FileNotFoundError(f"错误：未找到 {gts_path} 目录")
        
        # 遍历 gts 目录下的 .npy 文件
        occ_files = sorted(glob.glob(os.path.join(gts_path, '*.npy'), recursive=False))
        
        # 检查是否有 .npy 文件
        if not occ_files:
            raise ValueError(f"错误：在 {gts_path} 目录中未找到任何 .npy 文件")
        
        print(f"找到 {len(occ_files)} 个 .npy 文件")
        
        # 存储当前数据集的信息列表
        dataset_infos = []
        
        # 遍历每个 .npy 文件
        for frame_idx, occ_file in enumerate(occ_files):
            print(f"处理帧 {frame_idx}: {occ_file}")
            
            # 构建相机信息字典
            cams = {}
            missing_cameras = []
            
            for camera_name in fisheye_cameras:
                image_filename = os.path.basename(occ_file).replace('.npy', '.jpg')
                image_path = os.path.join(root_path, 'Camera', camera_name, image_filename)
                
                # 检查图像是否存在
                if not os.path.exists(image_path):
                    missing_cameras.append(f"{camera_name} 图像")
                    print(f"警告：未找到 {image_path}")
                    sys.exit(1)
                
                # 从 sensor_info 中查找匹配的相机参数
                camera_info = next((cam for cam in sensor_info['sensors']['Cameras'] if cam['name'] == camera_name), None)
                if camera_info is None:
                    print(f"错误：未找到 {camera_name} 的相机参数，程序终止")
                    sys.exit(1)

                # 提取相机内参和畸变系数
                # print("Camera Info:", camera_info)  # 调试打印
                
                # 根据实际数据结构提取内参
                K = np.array(camera_info['intrinsic']['K'])
                D = np.array(camera_info['intrinsic']['D'])
                
                # 提取外参信息并构建 T_l2c 矩阵
                extrinsic = camera_info.get('extrinsic', {}).get('to_lidar_main', [])
                
                # 构建变换矩阵
                T_l2c = np.zeros((4, 4))
                
                # 平移部分
                T_l2c[:3, 3] = extrinsic[:3]
                
                # 四元数转旋转矩阵
                quat = extrinsic[3:]
                rotation_matrix = R.from_quat(quat).as_matrix()
                T_l2c[:3, :3] = rotation_matrix
                
                T_l2c[3, 3] = 1
                
                cams[camera_name] = {
                    'image_path': image_path,
                    'intrinsic': K.tolist(),
                    'distortion': D.tolist(),
                    'extrinsic': extrinsic,
                    'T_l2c': T_l2c.tolist()  # 添加 T_l2c 矩阵
                }
                
                # print(f"成功处理相机 {camera_name}")
            
            # 检查是否所有相机都已找到
            if len(cams) != len(fisheye_cameras):
                print(f"警告：未找到所有指定的鱼眼相机：{', '.join(missing_cameras)}")
                sys.exit(1)
            
            # 构建信息字典，尽量模仿 NuScenes 格式
            token = os.path.basename(occ_file).split('.')[0]
            info = {
                'lidar_path': None,  # 不再加载点云数据
                'occ_path': occ_file,  # 使用 occ_path
                'token': token,
                'lidar_token': token,  # 添加 lidar_token
                'scene_token': f'baidu_scene_{os.path.basename(root_path)}',  # 使用文件夹名作为场景标识
                'frame_idx': frame_idx,  # 添加帧索引
                
                # 处理前后帧信息
                'prev': occ_files[frame_idx-1] if frame_idx > 0 else None,
                'next': occ_files[frame_idx+1] if frame_idx < len(occ_files)-1 else None,
                
                'sweeps': [],  # 如果有多帧数据可以在这里添加
                'cams': cams,  # 使用收集的所有鱼眼相机信息
                
                # 添加一些默认的标定和位姿信息
                'lidar2ego_translation': [0, 0, 1.0],  # 默认高度为1米
                'lidar2ego_rotation': [1, 0, 0, 0],  # 单位四元数
                'ego2global_translation': [0, 0, 0],  # 原点
                'ego2global_rotation': [1, 0, 0, 0],  # 单位四元数
                
                'timestamp': frame_idx,  # 使用帧索引作为时间戳
                
                # 边界框和标签信息
                'gt_boxes': None,  # 不再从 .npy 文件加载点云数据
                'gt_names': ['car'],  # 默认为车辆
                'gt_velocity': np.zeros((1, 2)),  # 速度信息
                
                # 点云和有效性信息
                'num_lidar_pts': 0,
                'num_radar_pts': 0,  # 假设没有雷达数据
                'valid_flag': np.ones(1, dtype=bool),
                
                # 添加语义分割和遮挡信息的占位
                'lidarseg': None,
                
                # 添加 can_bus 信息的占位
                'can_bus': np.zeros(18)  # NuScenes 中 can_bus 是 18 维向量
            }
            
            dataset_infos.append(info)
        
        all_infos.append({
            'path': root_path,
            'infos': dataset_infos
        })
    
    # 选择验证集文件夹
    if val_folder is None:
        # 如果没有指定，随机选择一个
        import random
        val_dataset = random.choice(all_infos)
    else:
        # 根据文件夹名选择验证集
        val_dataset = next((dataset for dataset in all_infos if val_folder in dataset['path']), all_infos[0])
    
    # 分割训练集和验证集
    train_infos = []
    for dataset in all_infos:
        if dataset != val_dataset:
            train_infos.extend(dataset['infos'])
    val_infos = val_dataset['infos']
    
    # 保存训练集信息
    train_info_path = osp.join(out_path, f'{info_prefix}_infos_temporal_train.pkl')
    mmcv.dump({
        'infos': train_infos,
        'metadata': {'version': 'v1.0-trainval', 'val_folder': val_dataset['path']}
    }, train_info_path)
    
    # 保存验证集信息
    val_info_path = osp.join(out_path, f'{info_prefix}_infos_temporal_val.pkl')
    mmcv.dump({
        'infos': val_infos,
        'metadata': {'version': 'v1.0-trainval', 'val_folder': val_dataset['path']}
    }, val_info_path)
    
    print(f"总数据集数量：{len(all_infos)}")
    print(f"训练集样本数：{len(train_infos)}")
    print(f"验证集样本数：{len(val_infos)}")
    print(f"验证集文件夹：{val_dataset['path']}")
    print(f"训练集信息已保存到：{train_info_path}")
    print(f"验证集信息已保存到：{val_info_path}")

def main():
    # 定义数据集根目录和输出目录
    dataset_root = '/home/datasets/ads_largedata/apa_parking_data/baidu_occ_data/datasets'
    out_path = '/home/datasets/ads_largedata/apa_parking_data/baidu_occ_data/pkl_data'
    
    # 获取所有子文件夹
    all_folders = [
        os.path.join(dataset_root, folder) 
        for folder in os.listdir(dataset_root) 
        if os.path.isdir(os.path.join(dataset_root, folder))
    ]
    
    # 确保输出目录存在
    os.makedirs(out_path, exist_ok=True)
    
    # 转换数据集，使用当前文件夹作为验证集
    create_baidu_occ_infos(
        all_folders, 
        out_path, 
        val_folder=all_folders[-1]  # 选择最后一个文件夹作为验证集
    )

if __name__ == '__main__':
    main()
