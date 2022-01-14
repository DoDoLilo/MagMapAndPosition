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

mag_map = [[[1, 1], [1, 1], [1, 1]],
           [[1, 1], [-1, -1], [1, 1]],
           [[1, 1], [1, 1], [1, 1]]]
list = MMT.interpolation_to_fill(mag_map)
for p in list[1]:
    print(mag_map[p[0]][p[1]])
