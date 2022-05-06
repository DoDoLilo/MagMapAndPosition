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
# 低通滤波的程度，值越大滤波越强。整型，无单位。
EMD_FILTER_LEVEL = 3
INTER_RADIUS = 1
INTER_TIME_THR = 2
DELETE_LEVEL = 1
# ------------------------
BUFFER_DIS = 5
DOWN_SIP_DIS = BLOCK_SIZE


path_pdr_raw = ["../data/data_test/pdr/IMU-10-6-178.00767980919863 Pixel 6.csv.npy",
                "../data/data_test/data_to_building_map/IMU-10-6-178.00767980919863 Pixel 6_sync.csv"]

pdr_xy = np.load(path_pdr_raw[0])[:, 0:2]
data_all = MMT.get_data_from_csv(path_pdr_raw[1])
raw_mag = data_all[:, 21:24]
raw_ori = data_all[:, 18:21]
raw_xy = data_all[:, np.shape(data_all)[1] - 5:np.shape(data_all)[1] - 3]
PDR_xy_mag_ori = MMT.get_PDR_xy_mag_ori(pdr_xy, raw_mag, raw_ori)
pdr_data_mag = PDR_xy_mag_ori[:, 2:5]
pdr_data_ori = PDR_xy_mag_ori[:, 5:8]
pdr_data_xy = pdr_xy
# 并不在此时修改pdr_xy坐标，match_seq_list=多条匹配序列[?][?][x,y, mv, mh]
match_seq_list = MMT.samples_buffer(BUFFER_DIS, DOWN_SIP_DIS, pdr_data_ori, pdr_data_mag, pdr_data_xy,
                                    do_filter=True, lowpass_filter_level=EMD_FILTER_LEVEL)
map_xy_list = []
for i in range(0, len(match_seq_list)):
    match_seq = np.array(match_seq_list[i])
    map_xy_list.append(match_seq[:, 0:2])

final_xy = []
for map_xy in map_xy_list:
    for xy in map_xy:
        final_xy.append(xy)
final_xy = np.array(final_xy)
# MMT.paint_xy(pdr_xy)
# MMT.paint_xy(final_xy)
new_pdr_xy = np.array(MMT.transfer_axis_list(final_xy, [6.8, 2.4, math.radians(-100.)]))
MMT.paint_xy(new_pdr_xy, xy_range=[0, MAP_SIZE_X, 0, MAP_SIZE_Y])
MMT.change_axis(raw_xy, MOVE_X, MOVE_Y)
MMT.paint_xy(raw_xy, xy_range=[0, MAP_SIZE_X, 0, MAP_SIZE_Y])
print("First Points, PDR:\n{0}, \niLocator:\n{1}".format(new_pdr_xy[0:5], raw_xy[0:5]))


