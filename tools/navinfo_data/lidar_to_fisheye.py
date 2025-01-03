#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
didi可视化的质检视频,2d框改为lidar投影的伪3d 框
"""

import argparse
import json
import os
import pypcd4 as pypcd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from bbox3d import BBox3D
from scipy.spatial.transform import Rotation
import glob
from PIL import Image, ImageDraw
np.set_printoptions(suppress=True)

all_cam_list = [
    'fisheye_front',
    'fisheye_right',
    'fisheye_left',
    'fisheye_back',
    "back",
    "front_narrow",
    "front_wide",
    "left_back",
    "left_front",
    "right_back",
    "right_front"
]

fish_eye_cam_names = [
    'fisheye_front',
    'fisheye_right',
    'fisheye_left',
    'fisheye_back'
]

other_cam_names = [
    "back",
    "front_narrow",
    "front_wide",
    "left_back",
    "left_front",
    "right_back",
    "right_front"
]



def box_center_to_corner(box, lidar_velocity):
    """
    根据3d框中心点获取8个角点
    :return:
    """
    x, y, z = box[0], box[1], box[2]
    l, w, h = box[3], box[4], box[5]
    yaw = box[6]
    r = Rotation.from_euler('xyz', [0, 0, yaw], degrees=False)
    bboxes = [[-l / 2.0, +w / 2.0, -h / 2.0], [-l / 2.0, +w / 2.0, +h / 2.0], [-l / 2.0, -w / 2.0, +h / 2.0],
              [-l / 2.0, -w / 2.0, -h / 2.0], [+l / 2.0, +w / 2.0, -h / 2.0], [+l / 2.0, +w / 2.0, +h / 2.0],
              [+l / 2.0, -w / 2.0, +h / 2.0], [+l / 2.0, -w / 2.0, -h / 2.0], [0, 0, 0], [lidar_velocity[0]/10, lidar_velocity[1]/10, 0]]
    bboxes = np.array([np.dot(r.as_matrix(), np.array(b)) + np.array([x, y, z]) for b in bboxes])
    return bboxes


def project_points_to_img_for_fisheye(img, pcd, K, D, T_l2c):
    """

    :param img:
    :param pcd:
    :param K:
    :param D:
    :param T_l2c:
    :return:
    """
    points = pcd[:, :3]

    points = np.matmul(points, T_l2c[:3, :3].T) + T_l2c[:3, 3]

    mask = points[:, 2] > 0
    points = points[mask]
    if len(points) == 0:
        return points

    # projects with opencv project
    image_coords = cv2.fisheye.projectPoints(points.reshape(1, -1, 3),
                                             np.zeros(3), np.zeros(3), K,
                                             D)[0].reshape(-1, 2)
    # image_coords = cv2.fisheye.distortPoints(points, K, D)[0].reshape(-1, 2)
    mask = (image_coords[:, 0] >= 0) & (image_coords[:, 0] < img.shape[1]) & (
            image_coords[:, 1] >= 0) & (image_coords[:, 1] < img.shape[0])
    image_coords = image_coords.astype(np.int32)
    return image_coords[mask]

def project_points_to_img_for_fisheye_lyg(img, pcd, K, D, T_l2c):
    """

    :param img:
    :param pcd:
    :param K:
    :param D:
    :param T_l2c:
    :return:
    """
    points = pcd[:, :3]

    # points = np.matmul(points, T_l2c[:3, :3].T) + T_l2c[:3, 3]

    points_lidar = np.hstack([points, np.ones((points.shape[0], 1))])
    # ex_trans = T.reshape(3,-1)
    # rt = np.hstack((R, ex_trans))

    # b = np.array([0, 0, 0, 1])
    # rt = np.r_[rt,[b]]
    points_camera = np.dot(T_l2c, points_lidar.T)
    points = points_camera[0:3, :].T

    mask = points[:, 2] > 0
    points = points[mask]
    if len(points) == 0:
        return points

    # projects with opencv project
    image_coords = cv2.fisheye.projectPoints(points.reshape(1, -1, 3),
                                             np.zeros(3), np.zeros(3), K,
                                             D)[0].reshape(-1, 2)
    # image_coords = cv2.fisheye.distortPoints(points, K, D)[0].reshape(-1, 2)
    mask = (image_coords[:, 0] >= 0) & (image_coords[:, 0] < img.shape[1]) & (
            image_coords[:, 1] >= 0) & (image_coords[:, 1] < img.shape[0])
    image_coords = image_coords.astype(np.int32)
    return image_coords[mask]

def project_points_to_img_for_other_lyg(img, pcd, K, D, T_l2c):
    """

    :param img:
    :param pcd:
    :param K:
    :param D:
    :param T_l2c:
    :return:
    """
    points = pcd[:, :3]

    # points = np.matmul(points, T_l2c[:3, :3].T) + T_l2c[:3, 3]
    points_lidar = np.hstack([points, np.ones((points.shape[0], 1))])
    # ex_trans = T.reshape(3,-1)
    # rt = np.hstack((R, ex_trans))

    # b = np.array([0, 0, 0, 1])
    # rt = np.r_[rt,[b]]
    points_camera = np.dot(T_l2c, points_lidar.T)
    points = points_camera[0:3, :].T

    mask = points[:, 2] > 0
    points = points[mask]
    if len(points) == 0:
        return points

    image_coords = cv2.projectPoints(points.reshape(1, -1, 3),
                                             np.zeros(3), np.zeros(3), K,
                                             D)[0].reshape(-1, 2)
    mask = (image_coords[:, 0] >= 0) & (image_coords[:, 0] < img.shape[1]) & (
            image_coords[:, 1] >= 0) & (image_coords[:, 1] < img.shape[0])
    image_coords = image_coords.astype(np.int32)
    return image_coords[mask]

def project_points_to_img_for_other(img, pcd, K, D, T_l2c):
    """

    :param img:
    :param pcd:
    :param K:
    :param D:
    :param T_l2c:
    :return:
    """
    points = pcd[:, :3]

    points = np.matmul(points, T_l2c[:3, :3].T) + T_l2c[:3, 3]

    mask = points[:, 2] > 0
    points = points[mask]
    if len(points) == 0:
        return points

    image_coords = cv2.projectPoints(points.reshape(1, -1, 3),
                                             np.zeros(3), np.zeros(3), K,
                                             D)[0].reshape(-1, 2)
    mask = (image_coords[:, 0] >= 0) & (image_coords[:, 0] < img.shape[1]) & (
            image_coords[:, 1] >= 0) & (image_coords[:, 1] < img.shape[0])
    image_coords = image_coords.astype(np.int32)
    return image_coords[mask]


def get_file_header_index_map(header_line):
    """
    获取表头数据的index
    :param header_line: str:第一行数据（表头）
    :return: dict:数据名—index
    """
    header_name_list = header_line.split("\t")
    header_index_map = {}
    for index, header_name in enumerate(header_name_list):
        header_index_map[header_name] = index
    return header_index_map


def plot_bev_bbox(ax, bbox, ego_v, lidar_v, track_id, ego_phi, **kwargs):
    """bev bbox plot util function"""
    CUBE_BOTTOM_PLOT_CORNERS = [0, 1, 2, 3, 0]
    ax.plot(bbox.p[CUBE_BOTTOM_PLOT_CORNERS, 0], bbox.p[CUBE_BOTTOM_PLOT_CORNERS, 1], **kwargs)
    # 计算中心点
    center = np.mean(bbox.p[CUBE_BOTTOM_PLOT_CORNERS], axis=0)

    # 速度矢量
    ego_velocity = np.array(ego_v)
    lidar_velocity = np.array(lidar_v)

    # 绘制速度矢量
    # 这个参数控制箭头的缩放因子。scale = 1表示箭头的长度将直接与提供的向量大小相对应。如果需要调整箭头的大小，可以更改这个值（例如，scale = 0.1会使箭头变短）
    ax.quiver(center[0], center[1],
              lidar_velocity[0], lidar_velocity[1],
              angles='xy', scale_units='xy',
              scale=1, color='r', label='Velocity Vector')

    # ax.text(center[0], center[1], track_id, fontsize=7, color='blue')
    # 在中心点添加速度数字标注
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if xlim[0] <= center[0] <= xlim[1] and ylim[0] <= center[1] <= ylim[1]:
        # ax.text(center[0], center[1],
        #         f'{track_id}\n({round(ego_velocity[0], 2)}, {round(ego_velocity[1], 2)})\n{round(ego_phi, 2)}',
        #         va='center',
        #         fontsize=3, color='black')
        ax.text(center[0], center[1],
                f'{track_id}',
                va='center',
                fontsize=10, color='black')


def get_bev_image_with_boxes_from_pcd(pc: np.ndarray,
                                      bboxes: list,
                                      images_with_box_file_path: str,
                                      pc_range=(-80, 80, -80, 80),
                                      show=True):
    """
    根据pcd文件生成框标注在bev点云图片上
    :param pc:
    :param bboxes:
    :param images_with_box_file_path：
    :param pc_range:
    :param show:
    :return:
    """

    fig = plt.figure(figsize=(8, 8), dpi=90, tight_layout=True)
    fig.suptitle('BBoxes Visualization')
    axes = []

    ax = fig.add_subplot()
    axes.append(ax)

    # 默认所有点为黄色
    # num_points = len(pc)
    # colors = np.full(num_points, 'yellow')
    colors = pc[:, 3]

    ax.scatter(pc[:, 0], pc[:, 1], cmap='Spectral', c=colors, s=0.5, linewidth=0, marker='.')
    ax.set_xlim(pc_range[:2])
    ax.set_ylim(pc_range[2:])
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    for box_info in bboxes:
        box = box_info['box']
        ego_v = box_info['ego_v']
        lidar_v = box_info['lidar_v']
        track_id = box_info['track_id']
        ego_phi = box_info['ego_phi']
        plot_bev_bbox(ax, box, ego_v, lidar_v, track_id, ego_phi, color='b', linewidth=0.8)

    # 反转y轴
    ax.invert_yaxis()
    ax.invert_xaxis()

    if show:
        plt.savefig(images_with_box_file_path)


def get_merge_images(frame_name, files_with_boxes_dir_path, files_after_merge_dir_path, order):
    """
    将多路传感器融合
    :param frame_name:
    :param files_with_boxes_dir_path:
    :param files_after_merge_dir_path:
    :return:
    """
    # 创建一个空的输出图像
    # output_image = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 修改为所需的输出图像尺寸
    output_image = np.ones((1350, 1920, 3), dtype=np.uint8) * 255
    # image = mpimg.imread(os.path.join(files_with_boxes_dir_path, 'refine_lidar_bev', f'{frame_name}.jpg'))
    img = cv2.imread(os.path.join(files_with_boxes_dir_path, 'refine_lidar_bev', f'{frame_name}.jpg'))
    img = cv2.resize(img, (800, 800))
    x, y = 320, 275
    output_image[y:y + img.shape[0], x:x + img.shape[1]] = img

    camera_position = {
        'back': [1440, 810],
        'front_narrow': [1440, 270],
        'front_wide': [1440, 540],
        'left_back': [0, 1080],
        'left_front': [480, 1080],
        'right_back': [960, 1080],
        'right_front': [1440, 1080],
        'fisheye_front': [0, 0],
        'fisheye_right': [480, 0],
        'fisheye_left': [960, 0],
        'fisheye_back': [1440, 0]
    }
    for camera, position in camera_position.items():
        # image = mpimg.imread(os.path.join(files_with_boxes_dir_path, f'{camera}', f'{frame_name}.jpg'))
        img = cv2.imread(os.path.join(files_with_boxes_dir_path, f'{camera}', f'{frame_name}.jpg'))
        img = cv2.resize(img, (480, 270))
        x = position[0]
        y = position[1]
        output_image[y:y + img.shape[0], x:x + img.shape[1]] = img

    # 保存输出图像
    cv2.imwrite(os.path.join(files_after_merge_dir_path, f'image_{order}.jpg'), output_image)


def get_video(files_after_merge_dir_path, video_path):
    """
    根据不同帧生成视屏
    :param files_after_merge_dir_path:
    :param video_path:
    :return:
    """
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.system(f"ffmpeg -framerate 5 -i {files_after_merge_dir_path}/image_%3d.jpg -c:v libx264 -pix_fmt yuv420p {video_path}")


def draw_3d_box_in_cam(boxes_3d, sensor_info, camera_dir_path, files_with_boxes_dir_path, frame_name):
    """

    :param boxes_3d:
    :param sensor_info:
    :param camera_dir_path:
    :param files_with_boxes_dir_path:
    :return:
    """
    for cam_name in all_cam_list:
        cam_info = sensor_info[cam_name]

        K = np.array(cam_info['cam_intrinsic']['intrinsic'])
        D = np.array(cam_info['cam_intrinsic']['distortion'])
        T = np.zeros((4, 4))
        T[:3, :3] = cam_info['sensor2lidar_rotation']
        T[:3, 3] = cam_info['sensor2lidar_translation']
        T[3, 3] = 1
        lidar2cam_T = np.linalg.inv(T)

        img = cv2.imread(glob.glob(os.path.join(camera_dir_path, cam_name, f'{frame_name}.jpg'))[0])

        for lidar_box_info in boxes_3d:
            # lidar坐标系下
            # print(lidar_box_info['box'])
            x, y, z = lidar_box_info['box'].center[0], lidar_box_info['box'].center[1], lidar_box_info['box'].center[2]
            l, w, h = lidar_box_info['box'].length, lidar_box_info['box'].width, lidar_box_info['box'].height
            yaw = lidar_box_info['box'].rotation.as_euler('ZYX')[0]
            # print(x, y, z, l, w, h, yaw)

            # lidar坐标系下速度
            velo_x = lidar_box_info['lidar_v'][0]
            velo_y = lidar_box_info['lidar_v'][1]
            velo_z = 0
            lidar_velocity = np.array([velo_x, velo_y, velo_z])

            corners_3d = box_center_to_corner([x, y, z, l, w, h, yaw], lidar_velocity)

            if cam_name in fish_eye_cam_names:
                corners = project_points_to_img_for_fisheye_lyg(img, corners_3d, K, D, lidar2cam_T)

            if cam_name in other_cam_names:
                K = np.array(K).reshape(3, 3)
                corners = project_points_to_img_for_other_lyg(img, corners_3d, K, D, lidar2cam_T)

            if len(corners) == 10:
                # print(corners)
                center_in_2d = corners[8]
                v_end = corners[9]
                # 绘制框
                cv2.polylines(img, [corners[[0, 1, 2, 3], :]], True, (255, 0, 0), 2)
                cv2.polylines(img, [corners[[4, 5, 6, 7], :]], True, (255, 0, 0), 2)
                cv2.polylines(img, [corners[[1, 2, 6, 5], :]], True, (255, 0, 0), 2)
                cv2.polylines(img, [corners[[0, 3, 7, 4], :]], True, (255, 0, 0), 2)
                # 标注truck_id
                start_point = (int(center_in_2d[0]), int(center_in_2d[1]))
                track_id = f"{lidar_box_info['track_id']}"
                cv2.putText(img, track_id, start_point, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
                # # 绘制速度箭头
                # start_point = (int(center_in_2d[0]), int(center_in_2d[1]))
                # end_point = (int(v_end[0]), int(v_end[1]))
                # cv2.arrowedLine(img, start_point, end_point, (0, 0, 255), thickness=3, line_type=cv2.LINE_4,
                #                 shift=0, tipLength=0.2)
                # # 标注速度
                # velocity_text = f'vx:{round(velo_x, 2)}, vy:{round(velo_y, 2)}, phi:{round(yaw, 2)}'
                # cv2.putText(img, velocity_text, start_point, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                #             color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # 绘制矩形，厚度为 2
        # cv2.rectangle(img, (0, 0), (200, 50), (0, 0, 0), -1)
        cv2.putText(img, f"{cam_name}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                    color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        os.makedirs(os.path.join(files_with_boxes_dir_path, cam_name), exist_ok=True)
        # print(f"draw {os.path.join(files_with_boxes_dir_path, cam_name, frame_name + '.jpg')}")
        cv2.imwrite(os.path.join(files_with_boxes_dir_path, cam_name, frame_name + '.jpg'), img)


def get_one_video(clip_path, video_path):
    """
    :param clip_path:
    :param video_path:
    :return:
    """
    camera_list = all_cam_list
    refine_lidar_dir_path = os.path.join(clip_path, 'refine_lidars')
    camera_dir_path = os.path.join(clip_path, 'imgs')
    label_json_dir_path = os.path.join(clip_path, 'labels')

    # 加框后文件的存储位置
    # files_with_boxes_dir_path = 'data/baidu/final_data/final_data_1127/files_with_boxes'
    files_with_boxes_dir_path = 'tmp/files_with_boxes'
    # 拼接后文件的存储位置
    # files_after_merge_dir_path = 'data/baidu/final_data/final_data_1127/files_after_merge'
    files_after_merge_dir_path = 'tmp/files_after_merge'

    frame_name_list = sorted(list(map(lambda x: x.split('.')[0], os.listdir(refine_lidar_dir_path))))
    i = 1
    for frame_name in frame_name_list[:25]:
        print(f"start dealing frame_name : {frame_name}")

        # 获取框信息
        one_label_json_file_path = os.path.join(label_json_dir_path, f'{frame_name}.json')
        # 2d框
        boxes_3d = []

        with open(one_label_json_file_path, "r") as f:
            label_json = json.load(f)

            # 获取3d
            lidar_object_info_list = label_json['frame_info']['lidar_object_info']['lidar_object_info']
            # 获取lidar2ego
            lidar2ego = np.zeros((4, 4))
            lidar2ego[:3, :3] = label_json['sensor_info']['lidar_info']['lidar_main']['lidar2ego_rotation']
            lidar2ego[:3, 3] = label_json['sensor_info']['lidar_info']['lidar_main']['lidar2ego_translation']
            lidar2ego[3, 3] = 1
            ego2lidar = np.linalg.inv(lidar2ego)

            # 获取sensor_info
            sensor_info = label_json['sensor_info']['camera_info']

            # 获取ego2golbal
            ego2golbal_7 = label_json['odom']
            # print("odom")
            # print(ego2golbal_7)
            quaternion_xyzw = [float(ego2golbal_7['qx']),
                               float(ego2golbal_7['qy']),
                               float(ego2golbal_7['qz']),
                               float(ego2golbal_7['qw'])]

            ego2global_matrix = Rotation.from_quat(quaternion_xyzw).as_matrix()


            ego2global_matrix_info = np.zeros((4, 4))
            ego2global_matrix_info[:3, :3] = ego2global_matrix
            ego2global_matrix_info[:3, 3] = [float(ego2golbal_7['wx']),
                                             float(ego2golbal_7['wy']),
                                             float(ego2golbal_7['wz'])]
            ego2global_matrix_info[3, 3] = 1

            for lidar_object_info in lidar_object_info_list[2:]:
                track_id = lidar_object_info['track_id']
                # ego坐标系下xyz
                p = [
                    float(lidar_object_info['position']['x']),
                    float(lidar_object_info['position']['y']),
                    float(lidar_object_info['position']['z'])
                ]

                # lidar坐标系下xyz
                x, y, z = Rotation.from_matrix(ego2lidar[:3, :3]).apply(p) + np.array(ego2lidar[:3, 3])

                # global坐标系下xyz
                # print(Rotation.from_matrix(ego2global_matrix_info[:3, :3]).apply(p) + np.array(ego2global_matrix_info[:3, 3]))

                # x, y, z, length, width, height,
                box = BBox3D(
                    x,
                    y,
                    z,
                    float(lidar_object_info['size'][2]),
                    float(lidar_object_info['size'][0]),
                    float(lidar_object_info['size'][1])
                )
                # phi取lidar坐标系下的phi
                # ego下的euler信息
                phi = float(lidar_object_info['orientation']['phi'])
                theta = float(lidar_object_info['orientation']['theta'])
                psi = float(lidar_object_info['orientation']['psi'])
                ego_position_matrix = np.zeros((4, 4))
                ego_position_matrix[:3, :3] = Rotation.from_euler('ZYX', [phi, theta, psi]).as_matrix()
                ego_position_matrix[:3, 3] = np.array([x, y, z])
                ego_position_matrix[3, 3] = 1
                lidar_position_matrix = ego2lidar @ ego_position_matrix
                lidar_phi = Rotation.from_matrix(lidar_position_matrix[:3, :3]).as_euler('ZYX')[0]
                box.rotation = Rotation.from_euler('xyz', [0, 0, lidar_phi])

                # ego坐标系下
                ego_velo_x = lidar_object_info['spd_x']
                ego_velo_y = lidar_object_info['spd_y']
                velo_z = 0

                # lidar坐标系下
                ego_velocity = np.array([ego_velo_x, ego_velo_y, velo_z])
                ego2lidar_R_matrix = np.array(ego2lidar)[:3, :3]
                lidar_velocity = np.dot(ego2lidar_R_matrix, ego_velocity)
                velo_x = lidar_velocity[0]
                velo_y = lidar_velocity[1]

                # global坐标系下
                # global_velocity = np.dot(ego2global_matrix, ego_velocity)
                # velo_x = global_velocity[0]
                # velo_y = global_velocity[1]

                boxes_3d.append(
                    {
                        "box": box,
                        "lidar_v": [velo_x, velo_y],
                        "ego_v": [ego_velo_x, ego_velo_y],
                        "track_id": track_id,
                        "lidar_phi": lidar_phi,
                        "ego_phi": phi
                    }
                )

        one_pcd_file_path = os.path.join(refine_lidar_dir_path, f'{frame_name}.pcd')
        one_bev_images_with_boxes_file_path = os.path.join(files_with_boxes_dir_path, 'refine_lidar_bev', f'{frame_name}.jpg')
        os.makedirs(os.path.dirname(one_bev_images_with_boxes_file_path), exist_ok=True)
        pcd = pypcd.PointCloud.from_path(one_pcd_file_path)
        pc = np.array([list(item) for item in pcd.pc_data])
        get_bev_image_with_boxes_from_pcd(pc, boxes_3d, one_bev_images_with_boxes_file_path)

        draw_3d_box_in_cam(boxes_3d, sensor_info, camera_dir_path, files_with_boxes_dir_path, frame_name)

        # 融合 图片到一张图片上
        os.makedirs(files_after_merge_dir_path, exist_ok=True)
        if i < 10:
            order = f"00{i}"
        elif 10 <= i < 100:
            order = f"0{i}"
        else:
            order = f"{i}"
        get_merge_images(frame_name, files_with_boxes_dir_path, files_after_merge_dir_path, order)
        i += 1
        # return

    # 根据不同帧生成视屏
    get_video(files_after_merge_dir_path, video_path)

    # 清除暂存数据
    # os.system(f"rm -rf {files_with_boxes_dir_path} {files_after_merge_dir_path}")


def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="处理一些命令行参数。")

    # 添加参数
    parser.add_argument('--clip_dir_path', type=str, help='clip信息存放路径')
    parser.add_argument('--video_file_path', type=str, help='生成视频文件的路径')

    # 解析参数
    args = parser.parse_args()

    # if not (args.clip_dir_path and args.clip_dir_path):
    #     print("请确保输入 入参 --clip_dir_path 和 --video_file_path")
    #     return
    # clip_path = args.clip_dir_path
    # video_path = args.video_file_path

    # clip_path = "data/baidu/final_data/final_data_1119/B_2024-09-21-12-03-32_1a5bce95-c1cf-48a7-b914-e3ab71d9ff33zip"
    # video_path = "data/baidu/final_data/final_data_1119/video/output.mp4"
    # clip_path = '/nfs/ofs-meta-data2/bd_upload_data/obstacle/obstacle_20241111_baiduys01/2024-09-18-17-22-06_687dd682-26f7-4d00-9577-140b26aaf34dzip'
    # video_path = 'tmp/output.mp4'

    # clip_path = 'data/baidu/final_data/final_data_1122/A_2024-09-24-12-17-17_5fabb7d4-bf81-41cf-a329-11cbf45eedcdzip'
    # video_path = 'data/baidu/final_data/final_data_1122/output.mp4'

    clip_path = 'data/baidu/final_data/final_data_1127/B_2024-10-27-15-30-26_a696712b-53cf-48a6-8da5-be8a0230c25azip'
    # video_path = 'data/baidu/final_data/final_data_1127/output.mp4'
    video_path = 'tmp/output.mp4'

    # 一个clip生成一个视频文件
    get_one_video(clip_path, video_path)


if __name__ == "__main__":
    main()