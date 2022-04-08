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
import math

MOVE_X = 7
MOVE_Y = 8
MAP_SIZE_X = 8
MAP_SIZE_Y = 13
BLOCK_SIZE = 0.25
INTER_RADIUS = 1
INTER_TIME_THR = 2
DELETE_LEVEL = 1
# ------------------------
BUFFER_DIS = 5
DOWN_SIP_DIS = BLOCK_SIZE

file_paths_build_map = [
    "data/data_test/data_to_building_map/IMU-10-1-190.80648806940607 Pixel 6_sync.csv",
    "data/data_test/data_to_building_map/IMU-10-2-183.5307793202117 Pixel 6_sync.csv",
    "data/data_test/data_to_building_map/IMU-10-3-170.97105500171142 Pixel 6_sync.csv",
    "data/data_test/data_to_building_map/IMU-10-4-180.40767532222338 Pixel 6_sync.csv",
    "data/data_test/data_to_building_map/IMU-10-5-170.2125898151382 Pixel 6_sync.csv",
    "data/data_test/data_to_building_map/IMU-10-6-178.00767980919863 Pixel 6_sync.csv"
]
# mag_map = MMT.build_map_by_files(
#     file_paths=file_paths_build_map,
#     move_x=MOVE_X, move_y=MOVE_Y,
#     map_size_x=MAP_SIZE_X, map_size_y=MAP_SIZE_Y,
#     # time_thr=INTER_TIME_THR,
#     radius=INTER_RADIUS, block_size=BLOCK_SIZE,
#     delete_level=DELETE_LEVEL
# )

#
# file_path_test = "data/data_test/data_to_building_map/IMU-10-1-190.80648806940607 Pixel 6_sync.csv"
# data_all = MMT.get_data_from_csv(file_path_test)
# data_mag = data_all[:, 21:24]
# data_ori = data_all[:, 18:21]
# data_x_y = data_all[:, np.shape(data_all)[1] - 5:np.shape(data_all)[1] - 3]
#
# MMT.change_axis(data_x_y, move_x=MOVE_X, move_y=MOVE_Y)
# match_seq_arr = MMT.samples_buffer(BUFFER_DIS, DOWN_SIP_DIS, data_ori, data_mag, data_x_y)
# for seq in match_seq_arr:
#     test_arr = np.array(seq)
#     rast_mv_mh = MMT.build_rast_mv_mh(test_arr[:, 2:4], test_arr[:, 0:2], MAP_SIZE_X, MAP_SIZE_Y, BLOCK_SIZE)
#     MMT.paint_heat_map(rast_mv_mh, show_mv=False)


# MMT.paint_heat_map(np.array(x_y_list))

# path = "data/data_test/data_to_building_map/IMU-10-1-190.80648806940607 Pixel 6_sync.csv"
# data_all = MMT.get_data_from_csv(path)
# data_mag = data_all[:, 21:24]
# data_g = data_all[:, 24:27]
# data_ori = data_all[:, 18:21]
# data_x_y = data_all[:, np.shape(data_all)[1]-5:np.shape(data_all)[1]-3]

# # 地磁总强度，垂直、水平分量，
# data_magnitude = MMT.cal_magnitude(data_mag)
# arr_mv_mh = MMT.get_mag_hv_arr(data_ori, data_mag)
# emd滤波
# mv_filtered_emd = MMT.lowpass_emd(arr_mv_mh[:, 0], 4)
# # mh_filtered_emd = MMT.lowpass_emd(arr_mv_mh[:, 1], 4)
# magnitude_filtered_emd = MMT.lowpass_emd(data_magnitude, 3)
# MMT.paint_signal(magnitude_filtered_emd)
# # 坐标变换
# MMT.change_axis(data_x_y, MOVE_X, MOVE_Y)
# rast_mv_mh = MMT.build_rast_mv_mh(arr_mv_mh, data_x_y, MAP_SIZE_X, MAP_SIZE_Y, BLOCK_SIZE)
# MMT.paint_heat_map(rast_mv_mh)
# MMT.interpolation_to_fill(rast_mv_mh)
# MMT.paint_heat_map(rast_mv_mh)

# 坐标转换测试
# transfer=[-1, 1, math.radians(90)]
# print(MMT.transferXY(transfer, 1, 1))

# 残差平方和测试
# m1 = np.array([[1, 2], [0.5, 4]])
# m2 = np.array([[1, 2], [3, 4]])
# print(MMT.cal_loss(m1, m2))

# arr1 = np.array([[1],
#                  [2],
#                  [3]])
#
# print(MMT.cal_GaussNewton_increment(np.array([[[1, 1]], [[3, 4]]]), np.array([[5, 6, 7], [8, 9, 10]]),
#       np.array([2, 2]), np.array([1, 1])))

# arr1 = np.array([1,2,3,4,5,6])
# arr2 = np.array([7,8,9,10,11,12])
# arr3 = np.vstack((arr1,arr2))
# print(arr3)
# print(arr3.transpose())
# path_pdr_raw = ["data/data_test/pdr/IMU-10-1-190.80648806940607 Pixel 6.csv.npy",
#                 "data/data_test/data_to_building_map/IMU-10-1-190.80648806940607 Pixel 6_sync.csv"]
# pdr_xy = np.load(path_pdr_raw[0])[:, 0:2]
# for d in pdr_xy:
#     print(d)
# data_all = MMT.get_data_from_csv(path_pdr_raw[1])
# raw_mag = data_all[:, 21:24]
# raw_ori = data_all[:, 18:21]
# raw_xy = data_all[:, np.shape(data_all)[1]-5:np.shape(data_all)[1]-3]
# PDR_xy_mag_ori = MMT.get_PDR_xy_mag_ori(pdr_xy, raw_mag, raw_ori)
#
# new_pdr_xy = np.array(MMT.transfer_axis_list(pdr_xy, [6.682912146175308, 1.7343581967157702, math.radians(-101.24301981875861)]))
# MMT.change_axis(raw_xy, MOVE_X, MOVE_Y)
# xy_range=[0, MAP_SIZE_X, 0, MAP_SIZE_Y]
# # MMT.paint_xy(PDR_xy_mag_ori[:, 0:2], xy_range=xy_range)
# MMT.paint_xy(new_pdr_xy, xy_range=xy_range)
# MMT.paint_xy(raw_xy, xy_range=xy_range)

path_pdr_raw = ["data/data_test/pdr/IMU-1-1-191.0820588816594 Pixel 3a.csv.npy",
                "data/data_test/data_server_room/IMU-1-1-191.0820588816594 Pixel 3a_sync.csv"]
pdr_xy = np.load(path_pdr_raw[0])[:, 0:2]
data_all = MMT.get_data_from_csv(path_pdr_raw[1])
raw_mag = data_all[:, 21:24]
raw_ori = data_all[:, 18:21]
PDR_xy_mag_ori = MMT.get_PDR_xy_mag_ori(pdr_xy, raw_mag, raw_ori)
pdr_data_mag = PDR_xy_mag_ori[:, 2:5]
pdr_data_ori = PDR_xy_mag_ori[:, 5:8]
pdr_data_xy = pdr_xy
# 并不在此时修改pdr_xy坐标，match_seq_list=多条匹配序列[?][?][x,y, mv, mh]
match_seq_list = MMT.samples_buffer(BUFFER_DIS, DOWN_SIP_DIS, pdr_data_ori, pdr_data_mag, pdr_data_xy,
                                    do_filter=True)
map_xy_list = []
for i in range(0, len(match_seq_list)):
    match_seq = np.array(match_seq_list[i])
    map_xy_list.append(match_seq[:, 0:2])

final_xy = []
for map_xy in map_xy_list:
    for xy in map_xy:
        final_xy.append(xy)
final_xy = np.array(final_xy)
MMT.paint_xy(pdr_xy)
MMT.paint_xy(final_xy)
new_pdr_xy = np.array(MMT.transfer_axis_list(final_xy, [6.682912146175308, 1.7343581967157702, math.radians(-88.)]))
MMT.paint_xy(new_pdr_xy, xy_range=[0, MAP_SIZE_X, 0, MAP_SIZE_Y])