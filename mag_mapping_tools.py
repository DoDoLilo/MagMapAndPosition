import math
import numpy as np
from scipy import signal
from PyEMD import EMD
from dtaidistance import dtw
import matplotlib.pyplot as plt


# 读csv文件所有列，返回指定列：输入：文件路径，返回：指定列数组numpy.ndarray
def get_data_from_csv(path, row_start, row_end):
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


# 对地磁强度进行指定factor的指定次数的下采样，输入：一维磁强数组，factor，下采样次数，返回：处理后的一维磁强数组
def down_sampling_by_mean(data, factor):
    data_down = []
    # 舍弃最末尾无法凑成一个Factor的数据
    for i in range(0, len(data) - factor + 1, factor):
        data_down.append(np.mean(data[i:i + factor]))
    return np.array(data_down)


# 对输入的一维数组进行绘制查看波形
def paint_signal(data_signal, title='signal'):
    plt.figure(figsize=(15, 5))
    x = range(0, len(data_signal))
    plt.title(label=title, loc='center')
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
    mv = abs(-math.sin(pitch)*mag[0]+math.sin(roll)*math.cos(pitch)*mag[1]+math.cos(roll)*math.cos(pitch)*mag[2])
    mh = math.sqrt(mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2] - mv * mv)
    return mv, mh


def get_mag_hv_arr(arr_ori, arr_mag):
    list_mv_mh = []
    for i in range(0, len(arr_ori)):
        list_mv_mh.append(get_mag_hv_2(arr_ori[i], arr_mag[i]))
    return np.array(list_mv_mh)


# 用论文方法和ori替代方案计算mv,mh

# 注意使用dtw时，数组前后加0
