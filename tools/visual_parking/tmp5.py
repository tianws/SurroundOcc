'''
Author: tianwenshan tianwenshan@navinfo.com
Date: 2025-01-08 16:27:05
LastEditors: tianwenshan tianwenshan@navinfo.com
LastEditTime: 2025-01-10 18:10:13
FilePath: /ori_code/tmp5.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import os
# 设置视频文件的输出路径、编码格式、帧率、尺寸等
import numpy as np

output_path = 'occupancy_pred2.avi'  # 输出视频文件名
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编解码器
fps = 15  # 帧率
frame_width = 2430  # 视频宽度
frame_height = 1080  # 视频高度
# 创建 VideoWriter 对象
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

root = "tmp4/"
path_list = os.listdir(root)
path_list.sort()
for pp in path_list:
    img = cv2.imread(root + pp)
    out.write(img)

out.release()

