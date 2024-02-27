import numpy as np
import mag_mapping_tools as MMT
import paint_tools as PT

# 1层函数
# 输入：其中一维（mv或mh）磁场指纹库数组，尺度范围（用具体值替代）s
# 输出：多尺度特征数组
# 实现：遍历全部ij数组，计算所有尺度，返回多维数组，TODO 存储下来
def get_all_area_features(map, s_num):
    rows = len(map)
    cols = len(map[0])

    map_fc = np.empty((rows, cols))
    map_td = np.empty((rows, cols))
    map_xgx = np.empty((rows, cols))
    for i in range(0, rows):
        for j in range(0, cols):
            if map[i][j] == -1:
                # -1的地方就不算了
                map_fc[i][j] = 0
                map_td[i][j] = 0
                map_xgx[i][j] = 0
                continue

            data = get_area_1Ddata(map, i, j, s_num)
            if len(data) < 3:
                continue
            map_fc[i][j] = area_fc(data)
            map_td[i][j] = area_td(data)
            map_xgx[i][j] = area_xgx(data)

    return map_fc, map_td, map_xgx

# 2层函数
# 遍历给定ij与尺度的方形区域，做成1维数组
def get_area_1Ddata(map, i, j, s_num):
    rows = len(map)
    cols = len(map[0])

    data = []
    for ti in range(i - round(s_num / 2), i + round(s_num / 2)):
        if ti < 0 or ti >= rows:
            continue
        for tj in range(j - round(s_num / 2), j + round(s_num / 2)):
            if tj < 0 or tj >= cols:
                continue
            if map[ti][tj] == -1:
                continue
            data.append(map[ti][tj])
    return data


# 3层函数
# 计算1维数组对应的区域特征
# 区域方差
def area_fc(data):
    n = len(data)
    mean = sum(data) / n
    f = 0
    for d in data:
        f += (d - mean) ** 2
    return f / n


# 区域梯度
def area_td(data):
    n = len(data)
    f = 0
    for i in range(0, n):
        if i == 0:
            f += (data[i + 1] - data[i]) / 0.3
            continue
        if i == n - 1:
            f += (data[i] - data[i - 1]) / 0.3
            continue
        f += (data[i + 1] - data[i - 1]) / 0.6
    return f / n


# 区域相关性
def area_xgx(data):
    n = len(data)
    mean = sum(data) / n
    f = 0
    for i in range(1, n):
        f += (data[i - 1] - mean) * (data[i] - mean)

    return f / ((n - 1) * area_fc(data))

# 序列特征------------------------------------------------------------
# 输入：磁场匹配序列[n][mv]or[n][mh]的某一尺度窗口子序列段
# 返回：该尺度窗口的单点特征
def seq_fc(sub_seq):
    n = len(sub_seq)
    mean = sum(sub_seq) / n

    t = 0
    for d in sub_seq:
        t += (d-mean)**2
    return t/n

# TODO min len(sub_seq) = 2
def seq_td(sub_seq):
    n = len(sub_seq)
    t = 0
    for i in range(0, n):
        if i == 0:
            t += (sub_seq[i + 1] - sub_seq[i]) / 0.3
            continue
        if i == n - 1:
            t += (sub_seq[i] - sub_seq[i - 1]) / 0.3
            continue
        t += (sub_seq[i + 1] - sub_seq[i - 1]) / 0.6

    return t/n

def seq_xgx(sub_seq):
    n = len(sub_seq)
    mean = sum(sub_seq) / n
    f = 0
    for i in range(1, n):
        f += (sub_seq[i - 1] - mean) * (sub_seq[i] - mean)

    return f / ((n - 1) * seq_fc(sub_seq))

def get_seq_sub_seq(seq, i, s_num):
    # 获取序列第i个位置前后半个s_num的子序列，s_num最小为3，len(sub_seq)最小为2
    s_num = 3 if s_num < 3 else s_num
    n = len(seq)
    if (i-int(s_num/2)) < 0:
        return seq[0:round(s_num/2)]
    if (i+int(s_num/2)) > n-1:
        return seq[n-round(s_num/2):n]

    return seq[i-int(s_num/2):i+(s_num-int(s_num/2))]

def get_seq_feature_seqs(seq, s_num):
    fc_seq = []
    td_seq = []
    xgx_seq = []
    for i in range(0, len(seq)):
        sub_seq = get_seq_sub_seq(seq, i, s_num)
        fc_seq.append(seq_fc(sub_seq))
        td_seq.append(seq_td(sub_seq))
        xgx_seq.append(seq_xgx(sub_seq))

    return fc_seq, td_seq, xgx_seq

# 地磁指纹库文件，[0]为mv.csv，[1]为mh.csv
# PATH_MAG_MAP = [
#     "../data/InfCenter server room/mag_map/map_F1_2_3_4_B_0.3_deleted/mv_qiu_2d.csv",
#     "../data/InfCenter server room/mag_map/map_F1_2_3_4_B_0.3_deleted/mh_qiu_2d.csv"
# ]

PATH_MAG_MAP = ['../data/XingHu hall 8F test/mag_map/map_F1_2_B_0.3_full/mv_qiu_2d.csv',
                '../data/XingHu hall 8F test/mag_map/map_F1_2_B_0.3_full/mh_qiu_2d.csv']
# 0层函数
# 读取给的mv.csv mh.csv文件，计算所有区域特征的数组，保存结果为csv文件
if __name__ == '__main__':
    # mag_map[i][j] = [mv, mh]
    mag_map = MMT.rebuild_map_from_mvh_files(PATH_MAG_MAP)

    # 尺度s米
    s = 5
    s_num = int(5/0.3)
    mv_map_fc, mv_map_td, mv_map_xgx = get_all_area_features(mag_map[:, :, 0], s_num)
    mh_map_fc, mh_map_td, mh_map_xgx = get_all_area_features(mag_map[:, :, 1], s_num)

    # 保存结果
    result_path = "results/mv_map_features/XingHu/s5/"
    np.savetxt(result_path+"fc.csv", mv_map_fc, delimiter=',')
    np.savetxt(result_path+"td.csv", mv_map_td, delimiter=',')
    np.savetxt(result_path+"xgx.csv", mv_map_xgx, delimiter=',')

    result_path = "results/mh_map_features/XingHu/s5/"
    np.savetxt(result_path+"fc.csv", mh_map_fc, delimiter=',')
    np.savetxt(result_path+"td.csv", mh_map_td, delimiter=',')
    np.savetxt(result_path+"xgx.csv", mh_map_xgx, delimiter=',')

