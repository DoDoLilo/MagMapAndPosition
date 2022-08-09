import mag_mapping_tools as MMT
import numpy as np
import os

# 坐标系位移（平移）
MOVE_X = 10.
MOVE_Y = 15.
# 地图坐标系大小 0-MAP_SIZE_X ，0-MAP_SIZE_Y
MAP_SIZE_X = 70.
MAP_SIZE_Y = 28.
# 地图地磁块大小
BLOCK_SIZE = 0.3
# 低通滤波的程度，值越大滤波越强。整型，无单位。
EMD_FILTER_LEVEL = 3
# 内插半径
INTER_RADIUS = 1
# 内插迭代次数上限，目前未使用。目前方案是内插满后再根据DELETE_LEVEL进行删除
# INTER_TIME_THR = 2
# 是否删除多余内插块
DELETE_EXTRA_BLOCKS = False
# 删除多余内插块的程度，越大删除的内插范围越大，可以为负值。
DELETE_LEVEL = -3

file_paths_build_map = [
    "data/XingHu hall 8F test/mag_map_build/1/IMU-88-1-155.58859472662894 Pixel 6_sync.csv",
    "data/XingHu hall 8F test/mag_map_build/2/IMU-88-2-181.05721265191326 Pixel 6_sync.csv",
    "data/XingHu hall 8F test/mag_map_build/3/IMU-88-3-276.6560168416993 Pixel 6_sync.csv",
    "data/XingHu hall 8F test/mag_map_build/4/IMU-88-4-201.15885553023264 Pixel 6_sync.csv"
]

# 构建文件夹
save_dir_name = 'map_F'
for file_path in file_paths_build_map:
    save_dir_name += os.path.basename(os.path.dirname(file_path)) + '_'
save_dir_name += "B_{0}".format(BLOCK_SIZE)
save_dir_path = 'data/XingHu hall 8F test/mag_map/' + save_dir_name
save_dir_path += '_full' if not DELETE_EXTRA_BLOCKS else '_deleted'
if not os.path.exists(save_dir_path):
    os.mkdir(save_dir_path)

mag_map = MMT.build_map_by_files_and_ilocator_xy(
    file_paths=file_paths_build_map,
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
print("Save files to:", save_dir_path)
