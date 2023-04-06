# 修改的mag_map_tools的函数也加个test前缀后加入这个文件中
import time
import math
import numpy as np


# 获取系统时间，毫秒
def get_time_in_millisecond():
    t = time.time()
    return int(round(t * 1000))


# 获取系统时间，秒
def get_time_in_second():
    return time.time()


# 计算iLocator xy和PDR xy的距离，用来判断PDR轨迹是否准确
# iLocator xy和PDR xy之间下标是均匀稀疏的
# 输入：iLocator xy，坐标系变换后的PDR xy，iLocator频率，PDR频率
# 返回：distance_and_PDR_iLocator_points[N][5]：保存距离和坐标
# 外层计算平均距离，则使用 np.mean(arr[:, 0], axis=0)即可。
def cal_distance_between_GT_and_PDR(gt_xy, PDR_xy, xy_align_size=10):
    if xy_align_size < 1:
        print("Wrong pdr_imu_align_size in cal_distance_between_iLocator_and_PDR")
        return None

    distance_and_PDR_GT_points = []

    for i in range(0, len(PDR_xy)):
        j = i * xy_align_size
        # NOTE: 全是深拷贝，非地址拷贝
        if j >= len(gt_xy):
            break
        distance_and_PDR_GT_points.append([cal_distance(PDR_xy[i], gt_xy[j]),
                                                 PDR_xy[i][0], PDR_xy[i][1],
                                                 gt_xy[j][0], gt_xy[j][1]])
    return np.array(distance_and_PDR_GT_points)


# 但PDR xy到MagPDR xy经过的根据距离的下采样，导致下标不是均匀稀疏，
# 所以需要在获取MagPDR xy的时候记录原PDR xy的下标，根据原PDR xy下标计算iLocator和MagPDR xy之间的距离
# 输入：iLocator xy，带下标信息的MagPDR xy[N][x, y, PDR_xy_index]，iLocator频率，PDR频率
# 返回：distance_and_PDR_iLocator_points[N][5]：保存距离和坐标
# 外层计算平均距离，则使用 np.mean(arr[:, 0], axis=0)即可。
def cal_distance_between_GT_and_MagPDR(gt_xy, map_xy_with_index, xy_align_size=10):
    if xy_align_size < 1:
        print("Wrong pdr_imu_align_size in cal_distance_between_iLocator_and_MagPDR")
        return None

    distance_and_PDR_GT_points = []
    for i in range(0, len(map_xy_with_index)):
        map_pdr_xy_index = map_xy_with_index[i][2]
        gt_xy_index = int(map_pdr_xy_index * xy_align_size)
        if gt_xy_index >= len(gt_xy):
            break
        # NOTE: 全是深拷贝，非地址拷贝
        distance_and_PDR_GT_points.append([cal_distance(map_xy_with_index[i], gt_xy[gt_xy_index]),
                                                 map_xy_with_index[i][0], map_xy_with_index[i][1],
                                                 gt_xy[gt_xy_index][0], gt_xy[gt_xy_index][1]])
    return np.array(distance_and_PDR_GT_points)


# 输入两点坐标，p1[x,y],p2[x,y]
# 返回两点距离
def cal_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 根据输入的点与真值的距离数组，计算1倍σ评估参数
# 返回一倍σ评估指标：65%的坐标与真值的距离是1m以内。
def cal_sigma_level(target_dis, dis_arr):
    target_num = 0
    for dis in dis_arr:
        if dis[0] <= target_dis:
            target_num += 1
    target_percent = target_num / len(dis_arr)
    sigma_percent = target_percent / 0.65
    return target_percent, sigma_percent
