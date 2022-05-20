import mag_mapping_tools as MMT
import numpy as np

# 坐标系位移（平移）
MOVE_X = 25.
MOVE_Y = 12.
# 地图坐标系大小 0-MAP_SIZE_X ，0-MAP_SIZE_Y
MAP_SIZE_X = 58.
MAP_SIZE_Y = 16.
# 地图地磁块大小
BLOCK_SIZE = 0.25
# 低通滤波的程度，值越大滤波越强。整型，无单位。
EMD_FILTER_LEVEL = 3
# 内插半径
INTER_RADIUS = 1
# 内插迭代次数上限，目前未使用。目前方案是内插满后再根据DELETE_LEVEL进行删除
# INTER_TIME_THR = 2
# 删除多余内插块的程度，越大删除的内插范围越大，可以为负值。
DELETE_LEVEL = -3

file_paths_build_map = [
    "data/data_test/data_to_building_map/one_floor_hall_hallway/IMU-519-1-4.247409484433081 Pixel 3a_sync.csv",
    "data/data_test/data_to_building_map/one_floor_hall_hallway/IMU-519-3-182.20603680993108 Pixel 3a_sync.csv"
]

mag_map = MMT.build_map_by_files(
    file_paths=file_paths_build_map,
    move_x=MOVE_X, move_y=MOVE_Y,
    map_size_x=MAP_SIZE_X, map_size_y=MAP_SIZE_Y,
    # time_thr=INTER_TIME_THR,
    radius=INTER_RADIUS, block_size=BLOCK_SIZE,
    delete=False,
    delete_level=DELETE_LEVEL,
    lowpass_filter_level=EMD_FILTER_LEVEL
)

# mag_map保存到 data/data_test/mag_map
# mag_map[i][j][mv][mh]
np.savetxt('data/data_test/mag_map/one_floor_hall_hallway/map_F1_3_B25_full/mv.csv', mag_map[:, :, 0], delimiter=',')
np.savetxt('data/data_test/mag_map/one_floor_hall_hallway/map_F1_3_B25_full/mh.csv', mag_map[:, :, 1], delimiter=',')

