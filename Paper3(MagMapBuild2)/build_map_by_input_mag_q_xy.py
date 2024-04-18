import math

import mag_mapping_tools as MMT
import numpy as np
import os

# 坐标系平移参数（m）
MOVE_X = 0.
MOVE_Y = 0.
# 地图坐标系大小 0-MAP_SIZE_X ，0-MAP_SIZE_Y（m）
# MAP_SIZE_X = 35.
# MAP_SIZE_Y = 20.
# 星湖
MAP_SIZE_X = 70.0
MAP_SIZE_Y = 28.0

# 地图地磁块大小
BLOCK_SIZE = 0.3
# 低通滤波的程度，值越大滤波越强。整型，无单位。
EMD_FILTER_LEVEL = 3
# 内插半径
INTER_RADIUS = 1
# 内插迭代次数上限，目前未使用。目前方案是内插满后再根据DELETE_LEVEL进行删除
# INTER_TIME_THR = 2
# 是否删除多余内插块
DELETE_EXTRA_BLOCKS = True
# 删除多余内插块的程度，越大删除的内插范围越大，可以为负值。
DELETE_LEVEL = 0


# file = '../Paper3(MagMapBuild2)/results/InfCenter/new_mag_q_gt_pdr.csv'
# file = '../Paper3(MagMapBuild2)/results/XingHu/new_mag_q_gt_pdr.csv'

# file = '../Paper3(MagMapBuild2)/results/InfCenter/new_mag_q_gt_pdr(CMM).csv'
file = '../Paper3(MagMapBuild2)/results/XingHu/new_mag_q_gt_pdr(CMM).csv'
# 构建指纹文件存储dir
save_dir_name = 'map_F'
# save_dir_path = '../Paper3(MagMapBuild2)/results/InfCenter/' + save_dir_name
save_dir_path = '../Paper3(MagMapBuild2)/results/XingHu/' + save_dir_name
save_dir_path += '_full' if not DELETE_EXTRA_BLOCKS else '_deleted'
if not os.path.exists(save_dir_path):
    os.mkdir(save_dir_path)

data_all = np.loadtxt(file, delimiter=',') # mag 0 1 2, quat 3 4 5 6, gt 7 8, pdr 9 10
mag_map = MMT.build_map_by_input_mag_q_xy(
    data_mag=data_all[:, 0:3], data_quat=data_all[:, 3:7], data_x_y=data_all[:, 9:11],
    move_x=MOVE_X, move_y=MOVE_Y,
    map_size_x=MAP_SIZE_X, map_size_y=MAP_SIZE_Y,
    # time_thr=INTER_TIME_THR,
    radius=INTER_RADIUS, block_size=BLOCK_SIZE,
    delete_extra_blocks=DELETE_EXTRA_BLOCKS,
    delete_level=DELETE_LEVEL,
    lowpass_filter_level=EMD_FILTER_LEVEL,
    fig_save_dir=save_dir_path
)

# 保存建库文件
np.savetxt(save_dir_path + '/mv_qiu_2d.csv', mag_map[:, :, 0], delimiter=',')
np.savetxt(save_dir_path + '/mh_qiu_2d.csv', mag_map[:, :, 1], delimiter=',')

rows = len(mag_map)
cols = len(mag_map[0])
mag_map_s = np.empty((rows, cols), dtype=float)

for i in range(0, rows):
    for j in range(0, cols):
        if mag_map[i][j][0] >= 0 and mag_map[i][j][1] >= 0:
            mag_map_s[i][j] = math.sqrt((mag_map[i][j][0])**2+(mag_map[i][j][1])**2)
        else:
            mag_map_s[i][j] = -1
np.savetxt(save_dir_path + '/ms(CMM).csv', mag_map_s, delimiter=',')

print("Save files to:", save_dir_path)