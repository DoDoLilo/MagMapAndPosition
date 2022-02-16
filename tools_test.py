# from dtaidistance import dtw
#
# s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
# s2 = [1, 1, 2, 1, 0, 1, 0, 0, 0, 0]
# distance = dtw.distance(s1, s2)
# print(distance)
# 所以使用DTW计算距离前先在开头末尾补上0，0，因为dtw开头末尾强制对齐
# import mag_mapping_tools as MMT
#
# data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# print(MMT.down_sampling_by_mean(data, 4))
# for i in 0,1:
#     print(i)
import mag_mapping_tools as MMT
import numpy as np

MOVE_X = 7
MOVE_Y = 8
MAP_SIZE_X = 8
MAP_SIZE_Y = 13
BLOCK_SIZE = 0.3
INTER_RADIUS = 1

file_paths = [
    "data/data_test/data_to_building_map/IMU-10-1-190.80648806940607 Pixel 6_sync.csv",
    "data/data_test/data_to_building_map/IMU-10-2-183.5307793202117 Pixel 6_sync.csv",
    "data/data_test/data_to_building_map/IMU-10-3-170.97105500171142 Pixel 6_sync.csv",
    "data/data_test/data_to_building_map/IMU-10-4-180.40767532222338 Pixel 6_sync.csv",
    "data/data_test/data_to_building_map/IMU-10-5-170.2125898151382 Pixel 6_sync.csv",
    "data/data_test/data_to_building_map/IMU-10-6-178.00767980919863 Pixel 6_sync.csv"
]
mag_map = MMT.build_map_by_files(file_paths, MOVE_X, MOVE_Y, MAP_SIZE_X, MAP_SIZE_Y, BLOCK_SIZE, INTER_RADIUS)



# path = "data/data_test/data_to_building_map/IMU-10-1-190.80648806940607 Pixel 6_sync.csv"
# data_all = MMT.get_data_from_csv(path)
# data_mag = data_all[:, 21:24]
# data_g = data_all[:, 24:27]
# data_ori = data_all[:, 18:21]
# data_x_y = data_all[:, np.shape(data_all)[1]-5:np.shape(data_all)[1]-3]
#
# # 地磁总强度，垂直、水平分量，
# data_magnitude = MMT.cal_magnitude(data_mag)
# arr_mv_mh = MMT.get_mag_hv_arr(data_ori, data_mag)
# # emd滤波
# # mv_filtered_emd = MMT.lowpass_emd(arr_mv_mh[:, 0], 4)
# # mh_filtered_emd = MMT.lowpass_emd(arr_mv_mh[:, 1], 4)
# # magnitude_filtered_emd = MMT.lowpass_emd(data_magnitude, 4)
# # 坐标变换
# MMT.change_axis(data_x_y, MOVE_X, MOVE_Y)
# rast_mv_mh = MMT.build_rast_mv_mh(arr_mv_mh, data_x_y, MAP_SIZE_X, MAP_SIZE_Y, BLOCK_SIZE)
# MMT.paint_heat_map(rast_mv_mh)
# MMT.interpolation_to_fill(rast_mv_mh)
# MMT.paint_heat_map(rast_mv_mh)





