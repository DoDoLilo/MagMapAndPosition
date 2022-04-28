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
# 还是需要这个函数的，就算PDR里有ATE/RTE，那也还需要一个和cal_distance_between_iLocator_and_MagPDR对比的相同类型的误差指标
# iLocator xy和PDR xy之间下标是均匀稀疏的
# 输入：iLocator xy，PDR xy，iLocator频率，PDR频率
# 返回：distance_and_PDR_iLocator_points[N][5]：保存距离和坐标
# 外层计算平均距离，则使用 np.mean(arr[:, 0], axis=0)即可。
def cal_distance_between_iLocator_and_PDR(iLocator_xy, PDR_xy, iLocator_frequency, PDR_frequency):
    if iLocator_frequency < PDR_frequency:
        print("Wrong frequency in cal_distance_between_iLocator_and_MagPDR: iLocator_frequency < PDR_frequency")
        return None
    if iLocator_frequency % PDR_frequency != 0:
        print("Wrong frequency in cal_distance_between_iLocator_and_MagPDR: iLocator_frequency % PDR_frequency != 0")
        return None

    step = iLocator_frequency // PDR_frequency
    distance_and_PDR_iLocator_points = []

    for i in range(0, len(PDR_xy)):
        j = i * step
        # NOTE: 全是深拷贝，非地址拷贝
        distance_and_PDR_iLocator_points.append([cal_distance(PDR_xy[i], iLocator_xy[j]),
                                                 PDR_xy[i][0], PDR_xy[i][1],
                                                 iLocator_xy[j][0], iLocator_xy[j][1]])
    return np.array(distance_and_PDR_iLocator_points)


# 但PDR xy到MagPDR xy经过的根据距离的下采样，导致下标不是均匀稀疏，
# 所以需要在获取MagPDR xy的时候记录原PDR xy的下标，根据原PDR xy下标计算iLocator和MagPDR xy之间的距离
# 输入：iLocator xy，带下标信息的MagPDR xy[N][x, y, PDR_xy_index]，iLocator频率，PDR频率
# 返回：distance_and_PDR_iLocator_points[N][5]：保存距离和坐标
# 外层计算平均距离，则使用 np.mean(arr[:, 0], axis=0)即可。
def cal_distance_between_iLocator_and_MagPDR(iLocator_xy, MagPDR_xy, iLocator_frequency, PDR_frequency):
    if iLocator_frequency < PDR_frequency:
        print("Wrong frequency in cal_distance_between_iLocator_and_MagPDR: iLocator_frequency < PDR_frequency")
        return None
    if iLocator_frequency % PDR_frequency != 0:
        print("Wrong frequency in cal_distance_between_iLocator_and_MagPDR: iLocator_frequency % PDR_frequency != 0")
        return None

    step = iLocator_frequency // PDR_frequency
    distance_and_PDR_iLocator_points = []

    for i in range(0, len(MagPDR_xy)):
        j = int(MagPDR_xy[i][2] * step)
        # NOTE: 全是深拷贝，非地址拷贝
        distance_and_PDR_iLocator_points.append([cal_distance(MagPDR_xy[i], iLocator_xy[j]),
                                                 MagPDR_xy[i][0], MagPDR_xy[i][1],
                                                 iLocator_xy[j][0], iLocator_xy[j][1]])
    return np.array(distance_and_PDR_iLocator_points)


# 输入两点坐标，p1[x,y],p2[x,y]
# 返回两点距离
def cal_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
