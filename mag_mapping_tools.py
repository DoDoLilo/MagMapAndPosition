import math
import numpy as np
from scipy import signal
from PyEMD import EMD
from dtaidistance import dtw
import matplotlib.pyplot as plt


# 读csv文件所有列，返回指定列：输入：文件路径，返回：指定列数组numpy.ndarray
def get_data_from_csv(path, row_start=-1, row_end=-1):
    if row_start == row_end == -1:
        return np.loadtxt(path, delimiter=",")
    if row_start < 0:
        row_start = 0
    return np.loadtxt(path, delimiter=",", usecols=range(row_start, row_end + 1))


# 对地磁三轴进行强度计算：输入：地磁三轴二维数组，返回：一维磁强数组numpy.ndarray
def cal_magnitude(data_mag):
    data_magnitude = []
    for d in data_mag:
        x = 0
        for m in d:
            x = x + m * m
        data_magnitude.append(np.sqrt(x))
    return np.array(data_magnitude)


# 对地磁强度进行低通滤波，输入：一维磁强数组、采样频率、截止频率，返回：滤波后的一维磁强数组
def lowpass_butter(data_magnitude, sample_frequency, cut_off_frequency):
    b, a = signal.butter(8, 2 * cut_off_frequency / sample_frequency, 'lowpass')
    return signal.filtfilt(b, a, data_magnitude)


def lowpass_emd(data_magnitude, cut_off):
    emd = EMD()
    imfs = emd(data_magnitude)
    data_filtered = np.sum(imfs[cut_off:, :], axis=0)
    return data_filtered


# 对地磁强度进行一次指定factor的下采样，输入：一维磁强数组，factor，返回：处理后的一维磁强数组
def down_sampling_by_mean(data, factor):
    data_down = []
    # 舍弃最末尾无法凑成一个Factor的数据
    for i in range(0, len(data) - factor + 1, factor):
        data_down.append(np.mean(data[i:i + factor]))
    return np.array(data_down)


# 对输入的一维数组进行绘制查看波形
def paint_signal(data_signal, title='signal', ylim=60):
    plt.figure(figsize=(15, 5))
    x = range(0, len(data_signal))
    plt.title(label=title, loc='center')
    plt.ylim(0, ylim)
    plt.plot(x, data_signal, label='line', color='g', linewidth=1.0, linestyle='-')
    plt.show()
    plt.close()
    return


# 获得向量v在向量g的法平面上的投影
# g一般为重力（需要取反），v为磁力
# def get_shadow(g, v):
#     v2 = [g[1] * v[2] - g[2] * v[1], g[2] * v[0] - g[0] * v[2], g[0] * v[1] - g[1] * v[0]]
#     ag = [v2[1] * g[2] - v2[2] * g[1], v2[2] * g[0] - v2[0] * g[2], v2[0] * g[1] - v2[1] * g[0]]
#     length_ag = math.sqrt(ag[0] * ag[0] + ag[1] * ag[1] + ag[2] * ag[2])
#     ans = [ag[0] / length_ag, ag[1] / length_ag, ag[2] / length_ag]
#     return ans


def cal_length(x):
    return math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])


# 根据投影到重力水平面的方法计算水平分量，再计算垂直分量
# def get_mag_hv_1(g, mag):
#     s = get_shadow(g, mag)
#     mh = cal_length(s)
#     mv = math.sqrt(mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2] - mh * mh)
#     return mv, mh


# 用论文方法和ori计算mv,mh
def get_mag_hv_2(ori, mag):
    pitch = math.radians(ori[1])
    roll = math.radians(ori[2])
    mv = abs(
        -math.sin(pitch) * mag[0] + math.sin(roll) * math.cos(pitch) * mag[1] + math.cos(roll) * math.cos(pitch) * mag[
            2])
    mh = math.sqrt(mag[0] ** 2 + mag[1] ** 2 + mag[2] ** 2 - mv ** 2)
    return mv, mh


def get_mag_hv_arr(arr_ori, arr_mag):
    list_mv_mh = []
    for i in range(0, len(arr_ori)):
        list_mv_mh.append(get_mag_hv_2(arr_ori[i], arr_mag[i]))
    return np.array(list_mv_mh)


# 用论文方法和ori替代方案计算mv,mh


# 注意使用dtw时，数组前后加0


# 内插填补
# mag_map[i][j][2]:栅格化、坐标正数化，后的三维数组，保存栅格i,j的平均磁强mv,mh ，=-1表示无效
# radius：填补所用的半径范围（m）
# block_size: 单个栅格的大小（m）
def interpolation_to_fill(mag_map, radius=1.0, block_size=0.3):
    # 1、通过radius和block_size计算出斜向、非斜向的最远块数(向下取整)
    slant_most = int(radius / math.sqrt(2 * block_size ** 2))
    not_slant_most = int(radius / block_size)
    # 2、对无效的mag_map[i][j],在这个范围内去找8个方向的有效栅格块，保存有效块的candidates[][3]{mv,mh,distance}
    slant_directions = [[-1, -1], [+1, +1], [-1, +1], [+1, -1]]
    not_slant_directions = [[-1, 0], [+1, 0], [0, -1], [0, +1]]
    empty_list = []
    succeed_list = []
    failed_list = []
    for i in range(0, len(mag_map)):
        for j in range(0, len(mag_map[0])):
            if mag_map[i][j][0] == -1:
                empty_list.append([i, j])
                # candidates保存候选项的 mv, mh, distance
                candidates = []
                # 4对方向，按对同步搜索
                for t in 0, 1:
                    # 斜向2对
                    find1 = search_by_direction_distance(mag_map, i, j, slant_directions[2 * t], slant_most)
                    find2 = search_by_direction_distance(mag_map, i, j, slant_directions[2 * t + 1], slant_most)
                    if find1[0] != -1 and find2[0] != -1:
                        candidates.append(find1)
                        candidates.append(find2)
                    # 水平向2对
                    find1 = search_by_direction_distance(mag_map, i, j, not_slant_directions[2 * t], not_slant_most)
                    find2 = search_by_direction_distance(mag_map, i, j, not_slant_directions[2 * t + 1], not_slant_most)
                    if find1[0] != -1 and find2[0] != -1:
                        candidates.append(find1)
                        candidates.append(find2)
                # 3、根据distance进行加权求出当前块的mv',mh'，替换原来的mag_map值
                if len(candidates) == 0:
                    failed_list.append([i, j])
                else:
                    print(candidates)
                    succeed_list.append([i, j])
                    mag_map[i][j][0], mag_map[i][j][1] = cal_weighted_average(candidates)

    return [empty_list, succeed_list, failed_list]


# 返回找到的：磁强mv,mh，距离distance
# 未找到则返回-1,-1,-1
def search_by_direction_distance(mag_map, i, j, direction, most_far):
    for distance in range(0, most_far):
        i = i + direction[0]
        j = j + direction[1]
        if -1 < i < len(mag_map) and -1 < j < len(mag_map[0]):
            if mag_map[i][j][0] != -1 and mag_map[i][j][1] != -1:
                return [mag_map[i][j][0], mag_map[i][j][1], distance + 1]
        else:
            break
    return [-1, -1, 0]


# 计算加权平均并返回
def cal_weighted_average(candidates):
    weight_all = 0
    mv_ave = 0
    mh_ave = 0
    for c in candidates:
        weight_all += c[2]
    for c in candidates:
        w = c[2] / weight_all
        mv_ave += w * c[0]
        mh_ave += w * c[1]
    return mv_ave, mh_ave


# ---------------------2022/2/14----------------------------------------
# 绘制二维坐标图
def paint_xy(arr_x_y):
    plt.scatter(arr_x_y[:, 0], arr_x_y[:, 1])
    plt.show()
    return


# 数组arr_mv_mh栅格化 block_size=0.3(m)
# 输入：arr_mv_mh[N][2] ， x_y_GT轨迹[N][2]，地图范围，块大小
# 输出：rast_mv_mh[x][y][mv][mh]
# 思路：①将机房固定分块；②落于同一块中的x_y_GT的对应mv_mh进行平均
# 实现：arr_1保存平均结果， arr_2保存落入当前块的点个数 n
def build_rast_mv_mh(arr_mv_mh, arr_xy_gt, map_size_x, map_size_y, block_size):
    # 先根据地图、块大小，计算块的个数，得到数组的长度。向上取整？
    shape = [math.ceil(map_size_x / block_size), math.ceil(map_size_y / block_size), 2]
    rast_mv_mh = np.empty(shape, dtype=float)
    # 不用考虑平均前求sum时候的溢出情况? sys.float_info.max（1.7976931348623157e + 308）
    # 创建的rast_mv_mh全置 -1
    for i in range(0, shape[0]):
        for j in (0, shape[1]):
            rast_mv_mh[i][j][0] = -1
            rast_mv_mh[i][j][1] = -1

    # 创建用来记录个数的数字，blocks_num
    blocks_num = np.zeros([shape[0], shape[1]], dtype=int)
    # 遍历arr_mv_mh\arr_xy_gt计算块总和
    for i in range(0, len(arr_xy_gt)):
        if arr_mv_mh[i][0] == -1 or arr_mv_mh[i][1] == -1:
            continue
        else:
            block_x = arr_xy_gt[i][0] // block_size
            block_y = arr_xy_gt[i][1] // block_size
            rast_mv_mh[block_x][block_y][0] += arr_mv_mh[i][0]
            rast_mv_mh[block_x][block_y][1] += arr_mv_mh[i][1]
            blocks_num[block_x][block_y] += 1
    # 平均
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if blocks_num[i][j] != 0:
                rast_mv_mh[i][j] /= blocks_num[i][j]
                # rast_mv_mh[i][j][0] /= blocks_num[i][j]
                # rast_mv_mh[i][j][1] /= blocks_num[i][j]

    return rast_mv_mh


# 数组mag_map坐标-坐标正数化：GT坐标 --> 全正数坐标
# 思路：每幅图的xy是固定一样的1P2，所以将坐标原点移动至左下角即可
# 实现：x,y直接加上一个固定整数，GT坐标边界{x轴-7m ~ 1m，y轴-8m ~ 5m} +7,+8 -->
#                             全正数坐标{x轴 0m ~ 8m，y轴 0m ~ 13m}
# 数组mag_map坐标-坐标反正数化：全正数坐标 --> GT坐标
# 思路：正数化的逆操作即可
def change_axis(arr_x_y_gt, move_x, move_y):
    for i in range(0, len(arr_x_y_gt)):
        arr_x_y_gt[i][0] += move_x
        arr_x_y_gt[i][1] += move_y
    return


# 手动挑出ilocator图片质量好的csv文件进行
# 函数建库：
# 输入：栅格化后的 mag_map[i][j][2]:三维数组，保存栅格i,j的平均磁强mv,mh ，=-1表示无效
# 输出：
def build_map_by_one_file():
    return


# 多个文件平均建库
def build_map_by_multi_files():
    return
