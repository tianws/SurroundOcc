'''
Author: tianwenshan tianwenshan@navinfo.com
Date: 2025-01-08 16:25:49
LastEditors: tianwenshan tianwenshan@navinfo.com
LastEditTime: 2025-01-09 21:48:29
FilePath: /ori_code/visual_parking_api.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os, sys
#import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
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
    fov_voxels = np.load(visual_path)

    # fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]
    fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]


    #figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))
    figure = mlab.figure(size=(1245, 1080), bgcolor=(1, 1, 1))
    # pdb.set_trace()
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        color = (0,1,0),
        # fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=0.012, #voxel_size - 0.05*voxel_size, # 0.01
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )
    mlab.points3d([pc_range[0]], [pc_range[1]], [pc_range[2]], scale_factor=0.05, color=(1, 0, 0), mode='cube')  
    mlab.outline(extent=[pc_range[0]+pc_range[0], pc_range[3]+pc_range[0], pc_range[1]+pc_range[1], pc_range[4]+pc_range[1], pc_range[2]+pc_range[2], pc_range[5]+pc_range[2]])
    mlab.axes(extent=[pc_range[0]+pc_range[0], pc_range[3]+pc_range[0], pc_range[1]+pc_range[1], pc_range[4]+pc_range[1], pc_range[2]+pc_range[2], pc_range[5]+pc_range[2]], 
              nb_labels=5)
    mlab.view(azimuth=45.0, elevation=45.0, distance=8, focalpoint=[pc_range[0], pc_range[1], pc_range[2]])
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors


    mlab.savefig(save_path)
    mlab.close()
