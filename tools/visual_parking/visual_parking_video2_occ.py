'''
Author: tianwenshan tianwenshan@navinfo.com
Date: 2025-01-08 16:25:17
LastEditors: tianwenshan tianwenshan@navinfo.com
LastEditTime: 2025-01-10 16:42:43
FilePath: /ori_code/visual_parking_video2_occ.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os, sys

# 在导入 mayavi 之前设置环境变量
os.environ['ETS_TOOLKIT'] = 'null'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

#import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
#import torch
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
import subprocess
from visual_parking_api import save_avi
# 设置视频文件的输出路径、编码格式、帧率、尺寸等
# output_path = 'tmp2/occupancy_gt.avi'  # 输出视频文件名
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编解码器
# fps = 3  # 帧率
# frame_width = 1280  # 视频宽度
# frame_height = 960  # 视频高度
# # 创建 VideoWriter 对象
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


root = "/home/tianws/keyidel/vis_occ/visual_dir_val/"
path_list = os.listdir(root)
path_list.sort()
for name in path_list:
    

        # cmd ="python /home/qianlei/disk/data/baidu/visual_parking_api.py " + '/home/qianlei/disk/data/baidu/tmp2/{}/{}.npy '.format(name, str(frame_list[0])) + '/home/qianlei/disk/data/baidu/tmp2/{}/{}.png'.format(name, str(frame_list[0]))
        # subprocess.run(cmd, shell=True)
        save_avi('{}/{}/pred.npy'.format(root, name), 'tmp_zitu/{}.png'.format(name))

        img = cv2.imread('tmp_zitu/{}_45_degree_view.png'.format(name))
        voxel_img = cv2.resize(img, (1080,1080))  
        img = cv2.imread(root + name +  "/0.jpg")
        img_front = cv2.resize(img, (675,540))  
        cv2.putText(img_front, 'Front', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        img = cv2.imread(root + name +  "/1.jpg")
        img_left = cv2.resize(img, (675,540))  
        cv2.putText(img_left, 'Left', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        img = cv2.imread(root + name +  "/2.jpg")
        img_right = cv2.resize(img, (675,540))  
        cv2.putText(img_right, 'Right', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        img = cv2.imread(root + name +  "/3.jpg")
        img_back = cv2.resize(img, (675,540))  
        cv2.putText(img_back, 'Back', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        img12 = np.concatenate([img_front, img_left],axis=0)
        img24 = np.concatenate([img_back, img_right],axis=0)
        img = np.concatenate([img12, img24, voxel_img],axis=1)

        cv2.imwrite('tmp4/{}.png'.format(name), img)

# out.release()  # 释放视频写入资源
cv2.destroyAllWindows()  # 关闭所有窗口
# mlab.show()
mlab.close()
