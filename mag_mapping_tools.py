import math
import numpy as np
from scipy import signal
from PyEMD import EMD
from dtaidistance import dtw
import matplotlib.pyplot as plt
import seaborn as sns


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


# cut_off: 舍弃的高频信号分量数量，该值越大，滤掉的高频信号越多
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
def get_mag_vh_2(ori, mag):
    pitch = math.radians(ori[1])
    roll = math.radians(ori[2])
    mv = abs(
        -math.sin(pitch) * mag[0] + math.sin(roll) * math.cos(pitch) * mag[1] + math.cos(roll) * math.cos(pitch) * mag[
            2])
    mh = math.sqrt(mag[0] ** 2 + mag[1] ** 2 + mag[2] ** 2 - mv ** 2)
    return mv, mh


# 输入：orientation[3], mag[3]
def get_mag_vh_arr(arr_ori, arr_mag):
    list_mv_mh = []
    for i in range(0, len(arr_ori)):
        list_mv_mh.append(get_mag_vh_2(arr_ori[i], arr_mag[i]))
    return np.array(list_mv_mh)


# 用论文方法和ori替代方案计算mv,mh？


# 内插填补
# 输入：mag_map[i][j][2]:栅格化、坐标正数化，后的三维数组，保存栅格i,j的平均磁强mv,mh ，=-1表示无效
#      radius：填补所用的半径范围（m）
#      block_size: 单个栅格的大小（m）
# 输出：空块、填充成功的块、失败的块
def interpolation_to_fill(mag_map, radius=1.0, block_size=0.3):
    # 0、使用copy数组以避免嵌套填补
    copy_mag_map = np.copy(mag_map)
    mask_list = []
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
                    find1 = search_by_direction_distance(copy_mag_map, i, j, slant_directions[2 * t], slant_most)
                    find2 = search_by_direction_distance(copy_mag_map, i, j, slant_directions[2 * t + 1], slant_most)
                    if find1[0] != -1 and find2[0] != -1:
                        candidates.append(find1)
                        candidates.append(find2)
                    # 水平向2对
                    find1 = search_by_direction_distance(copy_mag_map, i, j, not_slant_directions[2 * t],
                                                         not_slant_most)
                    find2 = search_by_direction_distance(copy_mag_map, i, j, not_slant_directions[2 * t + 1],
                                                         not_slant_most)
                    if find1[0] != -1 and find2[0] != -1:
                        candidates.append(find1)
                        candidates.append(find2)
                # 3、根据distance进行加权求出当前块的mv',mh'，替换原来的mag_map值
                if len(candidates) == 0:
                    failed_list.append([i, j])
                else:
                    succeed_list.append([i, j])
                    mag_map[i][j][0], mag_map[i][j][1] = cal_weighted_average(candidates)

    return [empty_list, succeed_list, failed_list]


# 返回找到的：磁强mv,mh，距离distance
# 未找到则返回-1,-1,0
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
        # 距离distance越大，权重越小
        weight_all += 1 / c[2]
    for c in candidates:
        w = (1 / c[2]) / weight_all
        mv_ave += w * c[0]
        mh_ave += w * c[1]
    return mv_ave, mh_ave


# ---------------------2022/2/14----------------------------------------
# 绘制二维坐标图
def paint_xy(arr_x_y, title=None, xy_range=None):
    if xy_range is not None:
        plt.figure(figsize=(xy_range[1] - xy_range[0], xy_range[3] - xy_range[2]))
        plt.xlim(xy_range[0], xy_range[1])
        plt.ylim(xy_range[2], xy_range[3])
    plt.title(title)
    plt.scatter(arr_x_y[:, 0], arr_x_y[:, 1])
    plt.show()
    return


# 数组arr_mv_mh栅格化 block_size=0.3(m)
# 输入：arr_mv_mh[N][2] ， x_y_GT轨迹[N][2]，地图范围，块大小
# 输出：rast_mv_mh[x][y][mv][mh]
# 思路：①将机房固定分块；②落于同一块中的x_y_GT的对应mv_mh进行平均
# 实现：arr_1保存平均结果， arr_2保存落入当前块的点个数 n
def build_rast_mv_mh(arr_mv_mh, arr_xy_gt, map_size_x, map_size_y, block_size):
    # 先根据地图、块大小，计算块的个数，得到数组的长度。向上取整
    shape = [math.ceil(map_size_x / block_size), math.ceil(map_size_y / block_size), 2]
    rast_mv_mh = np.empty(shape, dtype=float)
    # 不用考虑平均前求sum时候的溢出情况? sys.float_info.max（1.7976931348623157e + 308）
    # 创建的rast_mv_mh全置 -1
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            rast_mv_mh[i][j][0] = -1
            rast_mv_mh[i][j][1] = -1

    # 创建用来记录个数的数字，blocks_num
    blocks_num = np.zeros([shape[0], shape[1]], dtype=int)
    # 遍历arr_mv_mh\arr_xy_gt计算块总和
    for i in range(0, len(arr_xy_gt)):
        block_x = int(arr_xy_gt[i][0] // block_size)
        block_y = int(arr_xy_gt[i][1] // block_size)
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


# 手动挑出ilocator图片质量好的csv文件进行建库
# 输入：原始csv文件路径file_paths
# 输出：栅格化的地磁双分量数组
# 实现：①根据路径读取数组；②将多个文件的{地磁、方向、xyGT}连接为一个数组后进行建库。
def build_map_by_files(file_paths, move_x, move_y, map_size_x, map_size_y, time_thr=-1, radius=1, block_size=0.3,
                       delete=False, delete_level=0, lowpass_filter_level=3):
    if len(file_paths) == 0:
        return None
    data_all = get_data_from_csv(file_paths[0])
    data_mag = data_all[:, 21:24]
    data_ori = data_all[:, 18:21]
    data_x_y = data_all[:, np.shape(data_all)[1] - 5:np.shape(data_all)[1] - 3]

    for i in range(1, len(file_paths)):
        data_all = get_data_from_csv(file_paths[i])
        data_mag = np.vstack((data_mag, data_all[:, 21:24]))
        data_ori = np.vstack((data_ori, data_all[:, 18:21]))
        data_x_y = np.vstack((data_x_y, data_all[:, np.shape(data_all)[1] - 5:np.shape(data_all)[1] - 3]))
    # 地磁总强度，垂直、水平分量，
    arr_mv_mh = get_mag_vh_arr(data_ori, data_mag)
    # emd滤波，太慢了，不过是建库阶段，无所谓
    mv_filtered_emd = lowpass_emd(arr_mv_mh[:, 0], lowpass_filter_level)
    mh_filtered_emd = lowpass_emd(arr_mv_mh[:, 1], lowpass_filter_level)
    arr_mv_mh = np.vstack((mv_filtered_emd, mh_filtered_emd)).transpose()
    # 栅格化
    change_axis(data_x_y, move_x, move_y)
    rast_mv_mh = build_rast_mv_mh(arr_mv_mh, data_x_y, map_size_x, map_size_y, block_size)
    # 内插填补，绘制结果
    paint_heat_map(rast_mv_mh)
    rast_mv_mh_raw = rast_mv_mh.copy()
    inter_fill_completely(rast_mv_mh, time_thr, radius, block_size)
    # paint_heat_map(rast_mv_mh, fill_num)
    if delete:
        delete_far_blocks(rast_mv_mh_raw, rast_mv_mh, radius, block_size, delete_level)
    paint_heat_map(rast_mv_mh)
    return rast_mv_mh


# 根据块绘制栅格地磁强度图（热力图）
# cmap:YlOrRd
# 输入：[x][y][mv, mh]
def paint_heat_map(arr_mv_mh, num=0, show_mv=True, show_mh=True):
    if show_mv:
        plt.figure(figsize=(19, 10))
        plt.title('mag_vertical_' + str(num))
        sns.set(font_scale=0.8)
        sns.heatmap(arr_mv_mh[:, :, 0], cmap='YlOrRd', annot=True, fmt='.0f')
        # sns.heatmap(arr_mv_mh[:, :, 0], cmap='rainbow')
        plt.show()

    if show_mh:
        plt.figure(figsize=(19, 10))
        plt.title('mag_horizontal_' + str(num))
        sns.set(font_scale=0.8)
        sns.heatmap(arr_mv_mh[:, :, 1], cmap='YlOrRd', annot=True, fmt='.0f')
        # sns.heatmap(arr_mv_mh[:, :, 1], cmap='rainbow')
        plt.show()
    return


# 循环调用内插填补
# 输入:栅格化rast_mv_mh，循环次数上限time_threshold(-1则循环直到不存在新增块),填补半径radius,块大小block_size
# 输出：内插填补的次数 num
def inter_fill_completely(rast_mv_mh, time_thr=-1, radius=1, block_size=0.3):
    num = 0
    while True:
        num += 1
        if len(interpolation_to_fill(rast_mv_mh, radius, block_size)[1]) == 0 or num == time_thr:
            break
    return num


# 输入：内插前的栅格化数组rast_raw，内插后数组rast_inter，内插半径，块大小，删除程度(允许<0)
# delete_level: =0的时候，保留内插半径圆范围内的块（所以=0的时候也可能进行删除），>0的时候删除的更多，<0的时候保留范围会扩大！
# 输出：保留原始位置一个半径内的内插值后的数组
# 实现：1、copy rast_raw；2、遍历rast_raw，将要保留的位置在copy中置 1；3、修改rast_inter
def delete_far_blocks(rast_mv_mh_raw, rast_mv_mh_inter, radius, block_size, delete_level):
    # 1
    if len(rast_mv_mh_raw) == 0:
        return
    copy_rast_raw = np.copy(rast_mv_mh_raw)
    far_most = int(radius / math.sqrt(2 * block_size ** 2)) - delete_level
    if far_most < 1:
        far_most = 1
    # 2
    len_1 = len(rast_mv_mh_raw)
    len_2 = len(rast_mv_mh_raw[0])
    for i in range(0, len_1):
        for j in range(0, len_2):
            if rast_mv_mh_raw[i][j][0] != -1 or rast_mv_mh_raw[i][j][1] != -1:
                for b_i in range(i - far_most, i + far_most + 1):
                    for b_j in range(j - far_most, j + far_most + 1):
                        if -1 < b_i < len_1 and -1 < b_j < len_2:
                            copy_rast_raw[b_i][b_j][0] = 1
    # 3
    for i in range(0, len_1):
        for j in range(0, len_2):
            if copy_rast_raw[i][j][0] == -1:
                rast_mv_mh_inter[i][j][0] = -1
                rast_mv_mh_inter[i][j][1] = -1
    return


# --------------------------匹配阶段的算法----------------------------------------------------
# 采样缓冲池_非实时（也使用累积距离）包括稀疏采样处理
# 实现：1、计算mv,mh; 2、计算PDR累计距离
# 输入：缓冲池长度buffer_dis，下采样距离down_sip_dis，采集的测试序列[N][ori[3], mag[3], [x,y]]
# 正式测试匹配时输入的data_xy应该是PDR_x_y，而不是iLocator_xy，而且输出的是单条匹配序列
#  NOTE:此时data_mag\ori还需与PDR_x_y对齐（不需要精确对齐？）后再输入（且PDR_xy为20帧/s和iLocator/手机 200帧/s不同），
#  输出变为[PDR_x, PDR_y, aligned_mmv, aligned_mmh]
# 输出：多条匹配序列[?][M][x,y, mv, mh]，注意不要转成np.array，序列长度M不一样
def samples_buffer(buffer_dis, down_sip_dis, data_ori, data_mag, data_xy, do_filter=False, lowpass_filter_level=3):
    # 计算mv,mh分量,得到[N][mv, mh]
    arr_mv_mh = get_mag_vh_arr(data_ori, data_mag)
    # 滤波：对匹配序列中的地磁进行滤波，在下采样之前滤波，下采样之后太短了不能滤波？
    if do_filter:
        mv_filtered_emd = lowpass_emd(arr_mv_mh[:, 0], lowpass_filter_level)
        mh_filtered_emd = lowpass_emd(arr_mv_mh[:, 1], lowpass_filter_level)
        arr_mv_mh = np.vstack((mv_filtered_emd, mh_filtered_emd)).transpose()
    # for遍历data_xy，计算距离，达到down_sip_dis/2记录 i_mid，达到 down_sip_dis记录 i_end并计算down_sampling
    i_start = 0
    i_mid = -1
    dis_sum_temp = 0
    dis_sum_all = 0
    dis_mid = down_sip_dis / 2
    match_seq_list = []
    match_seq = []
    for i in range(1, len(data_xy)):
        dis_sum_temp += math.hypot(data_xy[i][0] - data_xy[i - 1][0], data_xy[i][1] - data_xy[i - 1][1])
        if dis_sum_temp >= down_sip_dis:
            # 当PDR仅两帧之间距离直接> down_sip_dis，则会导致连续进入该flow，使得 i_mid=-1
            if i_mid == -1:
                i_mid = i_start
            match_seq.append(down_sampling(i_start, i_mid, i, data_xy, arr_mv_mh))
            i_start = i
            i_mid = -1
            dis_sum_all += dis_sum_temp
            # dis_sum_temp -= down_sip_dis 这样写虽然很真实，但会导致dis_sum_temp与i_mid_xy含义不一致，
            # 之前的会影响后续的，而且在dis_sum_all+= dis_sum_temp的时候会重复加上之前未清0的距离
            dis_sum_temp = 0
        else:
            if i_mid == -1 and dis_sum_temp >= dis_mid:
                i_mid = i

        if dis_sum_all >= buffer_dis:
            # dis_sum_all -= buffer_dis 这样写，之前的会影响后续的
            dis_sum_all = 0
            match_seq_list.append(match_seq.copy())
            match_seq.clear()

    match_seq_list.append(match_seq)
    # Don't change to numpy array at this time, because the len(match_seq) is different
    return match_seq_list


# 对实时采集到的原始磁场序列进行重采样（以空间距离为尺度）
# NOTE:重采样前是要先进行垂直/水平分量计算的，因为和对应的orientation相关，而ori不能平均
# 实现：按距离下采样？直线距离/累积距离？
# --->累积距离：假设为0.3m，则0.15m处x,y的磁强 = 0.0 ~ 0.3m的磁强平均（坐标不能平均）
# 输入：下采样窗口下标start,mid,end，采集的匹配序列[x, y, mv, mh]
# 输出：稀疏的 位置-磁场 序列[x, y, mmv, mmh]
def down_sampling(i_start, i_mid, i_end, data_xy, arr_mv_mh):
    mmv = np.mean(arr_mv_mh[i_start:i_end, 0])
    mmh = np.mean(arr_mv_mh[i_start:i_end, 1])
    return data_xy[i_mid][0], data_xy[i_mid][1], mmv, mmh, i_mid


# 对建立的方格指纹库进行双线性插值法+偏微分
# 输入：处理完毕的地磁指纹库（栅格化、内插）mag_map_vh[x][y][mv, mh]，需要获取地磁插值、梯度的坐标 x,y ，block_size
# 输出：np.array[线性插值后的磁强[l_mv, l_mh], 2对梯度分量[[[[mv/x , mv/y]], [[mh/x, mh/y]]]] :[2][1×2]
# 实现：1、当前点 x,y得到4候选块下标 : [x//bs, x//bs +1] * [y//bs, y//bs +1]
#      2、候选块下标对应原始坐标到x,y的距离: x-x0=x%bs, x1-x=bs-x%bs, x1-x0=bs; y-y0=y%bs, y1-y=bs-y%bs, y1-y0=bs
# NOTE: 越界处理：1、若x//bs or y//bs越界，则直接返回-1，-1表示越界错误；2、若+1后越界，则用x//bs替代 x//bs +1
#    未越界但磁强不存在：1、x//bs or y//bs未越界但磁强不存在：返回-1，-1；2、+1后未越界但不存在，则用x//bs替代 x//bs +1
def get_linear_map_mvh_with_grad(mag_map, x, y, block_size):
    b_x = int(x // block_size)
    b_y = int(y // block_size)
    num_b_x = len(mag_map)
    num_b_y = 0 if num_b_x == 0 else len(mag_map[0])

    if -1 < b_x < num_b_x and -1 < b_y < num_b_y:
        m_p00 = mag_map[b_x][b_y]
        if m_p00[0] == -1 or m_p00[1] == -1:
            return np.array([-1, -1]), np.array([[[-1, -1]], [[-1, -1]]])

        m_p01 = mag_map[b_x][b_y + 1] if b_y + 1 < num_b_y and mag_map[b_x][b_y + 1][0] != -1 else m_p00
        m_p10 = mag_map[b_x + 1][b_y] if b_x + 1 < num_b_x and mag_map[b_x + 1][b_y][0] != -1 else m_p00
        m_p11 = mag_map[b_x + 1][b_y + 1] if b_x + 1 < num_b_x and b_y + 1 < num_b_y and mag_map[b_x + 1][b_y + 1][
            0] != -1 else m_p00

        x_x0 = x % block_size
        x1_x = block_size - x_x0
        x1_x0 = block_size
        y_y0 = y % block_size
        y1_y = block_size - y_y0
        y1_y0 = block_size
        m_pxy = y_y0 * (x_x0 * m_p11 + x1_x * m_p01) / (x1_x0 * y1_y0) + y1_y * (x_x0 * m_p10 + x1_x * m_p00) / (
                x1_x0 * y1_y0)
        grad_px = y_y0 * (m_p11 - m_p01) / (x1_x0 * y1_y0) + y1_y * (m_p10 - m_p00) / (x1_x0 * y1_y0)
        grad_py = x_x0 * (m_p11 - m_p10) / (x1_x0 * y1_y0) + x1_x * (m_p01 - m_p00) / (x1_x0 * y1_y0)

        return m_pxy, np.array([[[grad_px[0], grad_py[0]]], [[grad_px[1], grad_py[1]]]])

    print("Error: Out of Whole Map!")
    return np.array([-1, -1]), np.array([[[-1, -1]], [[-1, -1]]])


# 对get_linear_map_mvh_with_grad()的中选取插值候选块的思路进行优化，修改如下：
# 原来的方法，不管(x, y)落于块内部的哪里，其插值计算来源都是固定的那4块，而现在选取的候选块更合理。
# 1.对地磁块block_size * block_size 的区域再细分为 “田” 4小块；    |1 |2 |
# 2.给出待插值点(x, y)，其必定落在这4小块中的某一块；               |3 |4 |
# 3.若落在小块1，则候选块选取左上角的；落在2则选择右上角的... ...
# 4.而此时候选点P00,P01,P10,P11的坐标改为 候选块的中心点
def get_linear_map_mvh_with_grad_2(mag_map, x, y, block_size):
    b_x = int(x // block_size)
    b_y = int(y // block_size)
    num_b_x = len(mag_map)
    num_b_y = 0 if num_b_x == 0 else len(mag_map[0])

    if -1 < b_x < num_b_x and -1 < b_y < num_b_y and mag_map[b_x][b_y][0] != -1 and mag_map[b_x][b_y][1] != -1:
        # 根据x,y所处区域，获取候选块P00,P10,P01,P11的下标
        b_x0 = b_x - 1 if x % block_size <= block_size / 2 else b_x
        b_y0 = b_y - 1 if y % block_size <= block_size / 2 else b_y
        b_x1 = b_x0 + 1
        b_y1 = b_y0 + 1
        # 根据块下标获取块中心坐标：x_center = b_x * bs + bs/2
        half_bs = block_size / 2
        x0 = b_x0 * block_size + half_bs
        x1 = b_x1 * block_size + half_bs
        y0 = b_y0 * block_size + half_bs
        y1 = b_y1 * block_size + half_bs
        # 计算完中心坐标后，再对块下标越界修正，此时取到的地磁值都为非-1
        b_x0 = b_x0 if b_x0 >= 0 else b_x
        b_y0 = b_y0 if b_y0 >= 0 else b_y
        b_x1 = b_x1 if b_x1 < num_b_x else b_x
        b_y1 = b_y1 if b_y1 < num_b_y else b_y
        m_p00 = mag_map[b_x0][b_y0]
        m_p01 = mag_map[b_x0][b_y1]
        m_p10 = mag_map[b_x1][b_y0]
        m_p11 = mag_map[b_x1][b_y1]
        # 带入论文公式4.2、4.3、4.4计算地磁插值、地磁双方向梯度
        m_pxy = (y - y0) * ((x - x0) * m_p11 + (x1 - x) * m_p01) / ((x1 - x0) * (y1 - y0)) + \
                (y1 - y) * ((x - x0) * m_p10 + (x1 - x) * m_p00) / ((x1 - x0) * (y1 - y0))
        grad_px = (y - y0) * (m_p11 - m_p01) / ((x1 - x0) * (y1 - y0)) + (y1 - y) * (m_p10 - m_p00) / (
                (x1 - x0) * (y1 - y0))
        grad_py = (x - x0) * (m_p11 - m_p10) / ((x1 - x0) * (y1 - y0)) + (x1 - x) * (m_p01 - m_p00) / (
                (x1 - x0) * (y1 - y0))

        return m_pxy, np.array([[[grad_px[0], grad_py[0]]], [[grad_px[1], grad_py[1]]]])

    # print("Out of Mag Map!")
    return np.array([-1, -1]), np.array([[[-1, -1]], [[-1, -1]]])


# 坐标变换函数，论文公式4.7。求x1, y1关于transfer的偏导矩阵，论文公式4.13
# 输入：变换向量transfer=[_x, _y, _angle(弧度，绕PDR坐标系的(0, 0)逆时针)]，被变换坐标PDR(x0,y0)
# 输出：转换后的坐标 x1, y1，偏导矩阵 grad_xy(2*3)
# NOTE: t_angle是弧度！不是°。 degrees(x) : 将弧度转化为角度  radians(x) : 将角度转化为弧度。
def transfer_axis_with_grad(transfer, x0, y0):
    _x = transfer[0]
    _y = transfer[1]
    _angle = transfer[2]
    m_angle = np.array([[math.cos(_angle), -math.sin(_angle)],
                        [math.sin(_angle), math.cos(_angle)]])
    m_move = np.array([[_x],
                       [_y]])
    m_xy = np.array([[x0],
                     [y0]])
    ans = np.dot(m_angle, m_xy) + m_move
    grad_xy = np.array([[1, 0, -math.sin(_angle) * x0 - math.cos(_angle) * y0],
                        [0, 1, math.cos(_angle) * x0 - math.sin(_angle) * y0]])
    return ans[0][0], ans[1][0], grad_xy


# 输入：轨迹序列，转换向量
# 输出：转换后的轨迹list
def transfer_axis_list(pdr_xy, transfer):
    map_xy = []
    for xy in pdr_xy:
        x, y, grad = transfer_axis_with_grad(transfer, xy[0], xy[1])
        map_xy.append([x, y])
    return np.array(map_xy)


# 计算残差平方和，论文公式4.8 TODO:为什么不用平均残差？
# 输入：重采样的：PDR实时地磁序列M1[M][mv, mh]，Mag_Map地磁序列M2[M][mv, mh]
# 输出：残差平方和 S
# 实现：欧氏距离
def cal_loss(mag_arr_1, mag_arr_2):
    s = 0
    for m1, m2 in zip(mag_arr_1, mag_arr_2):
        s += (m1[0] - m2[0]) ** 2 + (m1[1] - m2[1]) ** 2
    return s


# 对cal_loss()的优化：
# 将地磁序列减去其均值后再计算残差，认为 相同形状、不同模值 的俩序列的匹配度应该更高(loss更小)。
def cal_loss_2(mag_arr_1, mag_arr_2):
    s = 0
    mv_arr_1 = remove_mean(mag_arr_1[:, 0])
    mh_arr_1 = remove_mean(mag_arr_1[:, 1])
    mv_arr_2 = remove_mean(mag_arr_2[:, 0])
    mh_arr_2 = remove_mean(mag_arr_2[:, 1])
    for mv_1, mh_1, mv_2, mh_2 in zip(mv_arr_1, mh_arr_1, mv_arr_2, mh_arr_2):
        s += (mv_1 - mv_2) ** 2 + (mh_1 - mh_2) ** 2
    return s


# 高斯牛顿迭代法，实现论文公式4.11 4.12
# 实现：仅计算两点之间的高斯牛顿迭代_transfer。 矩阵转置 arr.transpose，逆np.linalg.inv(arr)
# 输入：地磁梯度矩阵mag_map_grads[2][1×2]， 坐标梯度矩阵xy_grad[2×3]，重采样的PDR地磁mag_p[mv, mh]、指纹库的地磁mag_m[mv, mh]
# 输出：单点迭代结果向量_transfer[3×1] --> [1×3]
# NOTE：[1,2,3].shape不是[1×3]而是[0×3]， [[1,2,3]]才是[1×3] ！！！
def cal_GaussNewton_increment(mag_map_grad, xy_grad, mag_P, mag_M, matrix_H_inverse):
    m0 = np.dot(mag_map_grad[0], xy_grad)
    m1 = np.dot(mag_map_grad[1], xy_grad)
    _transfer = np.dot(matrix_H_inverse,
                       np.dot(m0.transpose(), mag_P[0] - mag_M[0]) + np.dot(m1.transpose(), mag_P[1] - mag_M[1]))
    return _transfer.transpose()[0]


# 计算高斯牛顿公式计算过程中的中间矩阵H
# 输入：地磁梯度矩阵mag_map_grads[2][1×2]， 坐标梯度矩阵xy_grad[2×3]
# 输出：计算后的中间矩阵H
def cal_matrix_H(mag_map_grad, xy_grad):
    m0 = np.dot(mag_map_grad[0], xy_grad)
    m1 = np.dot(mag_map_grad[1], xy_grad)
    matrix_H = np.dot(m0.transpose(), m0) + np.dot(m1.transpose(), m1)
    return matrix_H


# 注意全流程的时候进行-1检查！
# 匹配全流程：
# 输入：上一次迭代的transfer(i)向量，缓冲池给的稀疏序列sparse_PDR_Mag[M]，最终的地磁地图Mag_Map[i][j][mv, mh](存在-1)
# 输出：迭代越界失败标志，上一次迭代的transfer(i)对应残差平方和、坐标序列xy(i)， 迭代后的 transfer(i+1)，
def cal_new_transfer_and_last_loss_xy(transfer, sparse_PDR_mag, mag_map, block_size, step):
    # 1、将sparse_PDR_mag里的PDR x,y 根据 last_transfer 转换坐标并计算坐标梯度，得到 map_xy, xy_grad
    pdr_xy = sparse_PDR_mag[:, 0:2]
    pdr_mvh = sparse_PDR_mag[:, 2:4]
    map_xy = []
    xy_grads = []
    for xy in pdr_xy:
        x, y, grad = transfer_axis_with_grad(transfer, xy[0], xy[1])
        map_xy.append([x, y])
        xy_grads.append(grad)
    map_xy = np.array(map_xy)
    xy_grads = np.array(xy_grads)

    # 2、由map_xy到mag_map中获取 map_mvh[M][mv, mh], mag_map_grads[M][2][[1×2]]
    # 若获取到了-1，怎么办？表示到了地图外面，此种迭代方向错误-->提前结束迭代
    # NOTE: 此时map_mvh可能为空
    out_of_map = False
    map_mvh = []
    mag_map_grads = []
    for xy in map_xy:
        mvh, grad = get_linear_map_mvh_with_grad_2(mag_map, xy[0], xy[1], block_size)
        if mvh[0] == -1 or mvh[1] == -1:
            # print("The out point is:", xy, ", [", xy // block_size, "]")
            out_of_map = True
            break
        map_mvh.append(mvh)
        mag_map_grads.append(grad)
    map_mvh = np.array(map_mvh)
    mag_map_grads = np.array(mag_map_grads)

    # 3、计算残差平方和last_loss，如果out_of_map = True，则last_loss无效
    loss = cal_loss_2(pdr_mvh, map_mvh) if not out_of_map else None

    # 4、由cal_GaussNewton_increment(mag_map_grad, xy_grad, mag_P, mag_M, matrix_H_inverse) * step计算 _transfer
    new_transfer = transfer
    #    4.1 计算 H 及其 逆
    matrix_H = np.zeros([3, 3], dtype=float)
    for mag_map_grad, xy_grad in zip(mag_map_grads, xy_grads):
        matrix_H += cal_matrix_H(mag_map_grad, xy_grad)

    try:
        matrix_H_inverse = np.linalg.inv(matrix_H)
    except np.linalg.LinAlgError:
        matrix_H_inverse = np.linalg.pinv(matrix_H)
    #    4.2 计算 _transfer 及其总和
    for mag_map_grad, xy_grad, mag_P, mag_M in zip(mag_map_grads, xy_grads, pdr_mvh, map_mvh):
        _transfer = cal_GaussNewton_increment(mag_map_grad, xy_grad, mag_P, mag_M, matrix_H_inverse) * step
        new_transfer[0] += _transfer[0]
        new_transfer[1] += _transfer[1]
        new_transfer[2] += _transfer[2]

    # NOTE:如果out_of_map = True，则last_loss无效
    return out_of_map, loss, map_xy, new_transfer


# 输入：建图各种参数：图长宽、块大小，绘图轨迹序列(已经栅格化的)，迭代次数，
# 思路：创建和mag_map一样大小的二维空数组，在其中绘图，如果两次序列一样，则不绘图
def paint_iteration_results(map_size_x, map_size_y, block_size, last_xy, new_xy, num):
    # 先根据地图、块大小，计算块的个数，得到数组的长度。向上取整
    shape = [math.ceil(map_size_x / block_size), math.ceil(map_size_y / block_size)]
    map = np.zeros(shape, dtype=float)
    different = False
    for last, new in zip(last_xy, new_xy):
        if last[0] != new[0] or last[1] != new[1]:
            different = True
            break

    if different:
        for new in new_xy:
            map[int(new[0])][int(new[1])] = 1.
        plt.figure(figsize=(19, 10))
        plt.title('Iteration_' + str(num))
        sns.set(font_scale=0.8)
        sns.heatmap(map, cmap='YlOrRd', annot=False)
        plt.show()

    return


# ----------------PDR相关--------------------------------
# PDR window size = 200，simple frequency=200Hz/s，sliding window size=10
# 而PDR模型输出为该窗口的平均速度v，认为该1秒的窗口速度相同，每 10/200s = 0.05s输出一速度
# 得到的PDR_x,y是该速度v*0.05s，而我们的PDR程序第一个200窗口乘的是0.05s而不是1.0s
# 所以PDR_x,y[0..i..]对应的时间应该为 i*0.05s，对应的数据帧为 i*10
# 一个PDR_x,y坐标应该对应一个地磁值，和根据距离下采样时保持一致，该地磁值也应该用平均值！
# 和simpl_buffer里的一致，达到距离阈值，则返回该段均值和该段mid坐标

# ***所以PDR_x,y[i]对应的地磁 = mean( raw_mag[ i*10-5 : i*10 + 5] ); 不包括 +5

# 函数：获取pdr坐标对应的地磁、方向原始数据(非平均)
# 输入：pdr_xy[n1][2]=pdr模型输出的20Hz/s的xy序列, raw_mag[n2][3]=pdr对应的原始手机200Hz数据中的地磁，raw_ori[n2][3]=方向，
#      mean=False则使用单点地磁
# 输出：PDR_xy_mag_ori[2+3+3=8]矩阵，[x,y, mag[0],mag[1],mag[2], ori[0],ori[1],ori[2]]
def get_PDR_xy_mag_ori(pdr_xy, raw_mag, raw_ori, pdr_frequency=20, sampling_frequency=200):
    if sampling_frequency < pdr_frequency:
        print("Wrong frequency in get_PDR_xy_mag_ori: sampling_frequency < PDR_frequency")
        return None
    if sampling_frequency % pdr_frequency != 0:
        print("Wrong frequency in get_PDR_xy_mag_ori: sampling_frequency % PDR_frequency != 0")
        return None

    window_size = int(sampling_frequency / pdr_frequency)
    if window_size < 1:
        print("get_PDR_xy_mag_ori(): Wrong PDR&simpling frequency!")
        return None

    PDR_xy_mag_ori = []
    for i in range(0, len(pdr_xy)):
        raw_i = i * window_size
        # 越界则提前跳出循环，按理说正确的[pdr_frequency, simpling_frequency]不会有这种情况
        if raw_i >= len(raw_mag):
            print("get_PDR_xy_mag_ori(): Drop out in break!")
            break
        PDR_xy_mag_ori.append([pdr_xy[i][0], pdr_xy[i][1],
                               raw_mag[raw_i][0], raw_mag[raw_i][1], raw_mag[raw_i][2],
                               raw_ori[raw_i][0], raw_ori[raw_i][1], raw_ori[raw_i][2]])
    return np.array(PDR_xy_mag_ori)


# 输入的data_xy变为不同频率的PDR_xy的缓冲池函数，区别在于内置了对原始数据与PDRxy数据之间的对齐、平均的操作
# 输入：+ data_xy变为频率不同的PDR_xy，+ PDR轨迹频率pdr_frequency=20, ori/mag采样频率sampling_frequency=200
# 输出：多条匹配序列[?][M][x,y, mv, mh, PDRindex]，注意不要转成np.array，序列长度M不一样
# 平均：当频率为200与20时，PDR_x,y[i]对应的地磁 = mean( raw_data[ i*10-5 : i*10 + 5] ); 不包括 +5。
def samples_buffer_PDR(buffer_dis, down_sip_dis, data_ori, data_mag, PDR_xy, do_filter=False, lowpass_filter_level=3,
                       pdr_frequency=20, sampling_frequency=200):
    # 1.计算mv,mh分量,得到[N][mv, mh]
    arr_mv_mh = get_mag_vh_arr(data_ori, data_mag)
    # 2.滤波：对匹配序列中的地磁进行滤波，在下采样之前滤波，下采样之后太短了不能滤波
    if do_filter:
        mv_filtered_emd = lowpass_emd(arr_mv_mh[:, 0], lowpass_filter_level)
        mh_filtered_emd = lowpass_emd(arr_mv_mh[:, 1], lowpass_filter_level)
        arr_mv_mh = np.vstack((mv_filtered_emd, mh_filtered_emd)).transpose()

    # + 3.相比原来的samples_buffer()：
    #    需要将原始高频arr_mv_mh 变为和 低频data_xy(pdr_xy) 对齐、平均后的和data_xy(PDR_xy)同频的arr_mv_mh
    if sampling_frequency < pdr_frequency:
        print("Wrong frequency in samples_buffer_PDR: sampling_frequency < PDR_frequency")
        return None
    if sampling_frequency % pdr_frequency != 0:
        print("Wrong frequency in samples_buffer_PDR: sampling_frequency % PDR_frequency != 0")
        return None

    window_size = int(sampling_frequency / pdr_frequency)

    arr_mv_mh_pdr = []
    for i in range(0, len(PDR_xy)):
        raw_i = i * window_size
        window_start = raw_i - int(window_size / 2)
        window_end = raw_i + int(window_size / 2)
        if window_start < 0:
            window_start = 0
        if window_end > len(data_mag):
            window_end = len(data_mag)
        if window_start < window_end:
            # ！注意错误数组访问下标写法：arr_mv_mh[window_start:window_end][0]
            arr_mv_mh_pdr.append([np.mean(arr_mv_mh[window_start:window_end, 0]),
                                  np.mean(arr_mv_mh[window_start:window_end, 1])])
        else:
            break
    # paint_signal(arr_mv_mh[:, 1], "Before align with PDR")
    arr_mv_mh = np.array(arr_mv_mh_pdr)
    # paint_signal(arr_mv_mh[:, 1], "After align with PDR")
    # + 相比原来的samples_buffer()  End

    # 4. for遍历PDR_xy，计算距离，达到down_sip_dis/2记录 i_mid，达到 down_sip_dis记录 i_end并计算down_sampling
    i_start = 0
    i_mid = -1
    dis_sum_temp = 0
    dis_sum_all = 0
    dis_mid = down_sip_dis / 2
    match_seq_list = []
    match_seq = []
    # 使用len(arr_mv_mh)，而不是 len(PDR_xy)，因为经过 3. 后，前者有可能小于后者
    for i in range(1, len(arr_mv_mh)):
        dis_sum_temp += math.hypot(PDR_xy[i][0] - PDR_xy[i - 1][0], PDR_xy[i][1] - PDR_xy[i - 1][1])
        if dis_sum_temp >= down_sip_dis:
            # 当PDR仅两帧之间距离直接> down_sip_dis，则会导致连续进入该flow，使得 i_mid=-1
            if i_mid == -1:
                i_mid = i_start
            match_seq.append(down_sampling(i_start, i_mid, i, PDR_xy, arr_mv_mh))
            i_start = i
            i_mid = -1
            dis_sum_all += dis_sum_temp
            # dis_sum_temp -= down_sip_dis 这样写虽然很真实，但会导致dis_sum_temp与i_mid_xy含义不一致，
            # 之前的会影响后续的，而且在dis_sum_all+= dis_sum_temp的时候会重复加上之前未清0的距离
            dis_sum_temp = 0
        else:
            if i_mid == -1 and dis_sum_temp >= dis_mid:
                i_mid = i

        if dis_sum_all >= buffer_dis:
            # dis_sum_all -= buffer_dis 这样写，之前的会影响后续的
            dis_sum_all = 0
            match_seq_list.append(match_seq.copy())
            match_seq.clear()

    match_seq_list.append(match_seq)
    # Don't change to numpy array at this time, because the len(match_seq) is different
    return match_seq_list


# 候选transfer向量生成器：
# 输入一个向量，输出以该向量为中心，产生的一组用以小区域遍历的候选项。
# 输出的候选向量应该按由近到远的辐射顺序
# 可自定义参数: transfer各维度分别的增减粒度、增减次数
# 输入：初始向量original_transfer[△x, △y(米), △angle(弧度)]，
#       自定义参数config[2][3]=[[x,y(米),angle(弧度)增减粒度],[x,y,angle增减次数]]
# 输出：满足要求的候选transfer_candidates[m][3] （包括original_transfer）
def produce_transfer_candidates(original_transfer, config):
    transfer_candidates = []
    # transfer_candidates.append(original_transfer)
    x_candidates = []
    y_candidates = []
    angle_candidates = []
    # for循环增减次数次，向各个参数list中添加
    x_candidates.append(original_transfer[0])
    for tx in range(1, config[1][0]):
        x_candidates.append(original_transfer[0] + tx * config[0][0])
        x_candidates.append(original_transfer[0] - tx * config[0][0])

    y_candidates.append(original_transfer[1])
    for ty in range(1, config[1][1]):
        y_candidates.append(original_transfer[1] + ty * config[0][1])
        y_candidates.append(original_transfer[1] - ty * config[0][1])

    angle_candidates.append(original_transfer[2])
    for ta in range(1, config[1][2]):
        angle_candidates.append(original_transfer[2] + ta * config[0][2])
        angle_candidates.append(original_transfer[2] - ta * config[0][2])

    # 三重for循环将这些候选分量重组
    # for x in x_candidates:
    #     for y in y_candidates:
    #         for angle in angle_candidates:
    #             transfer_candidates.append([x, y, angle])

    # 将这些候选分量按变换距离升序重新排列组和。x,y,z_candidates内部已按升序
    # 所以，以(i_x + i_y + i_angle)作为新组合的下标，
    temp_list = []
    len_x = len(x_candidates)
    len_y = len(y_candidates)
    len_angle = len(angle_candidates)

    for i_t in range(0, len_x + len_y + len_angle):
        temp_list.append([])

    for i_x in range(0, len_x):
        for i_y in range(0, len_y):
            for i_angle in range(0, len_angle):
                temp_list[i_x + i_y + i_angle].append([x_candidates[i_x], y_candidates[i_y], angle_candidates[i_angle]])

    # 将temp_list中的范围升序结果提取到transfer_candidates中。NOTE：len(temp_list[i])不全相等！
    for temp in temp_list:
        for t in temp:
            transfer_candidates.append(t)

    return np.array(transfer_candidates)


# 输入：用于生成transfer_candidates：初始变换向量original_transfer，范围参数area_config，
#     用于高斯牛顿迭代的：待匹配序列match_seq，地磁地图mag_map，块大小block_size，迭代步长step，最大迭代次数max_iteration
#     用于筛选候选项的：目标损失target_loss
# 输出：最终选择的Transfer（当小范围寻找失败时，则返回original_transfer）。
# NOTE：注意 地址拷贝（浅拷贝） 和 值拷贝（深拷贝） 的问题。
# 如果遍历候选项的过程中找到了符合target_loss的，则提前返回结果
def produce_transfer_candidates_search_again(original_transfer, area_config,
                                             match_seq, mag_map, block_size, step, max_iteration,
                                             target_loss, break_advanced=False):
    # 1.生成小范围的所有transfer_candidates
    transfer_candidates = produce_transfer_candidates(original_transfer, area_config)

    # 2.遍历transfer_candidates进行高斯牛顿，结果添加到候选集candidates_loss_xy_tf
    candidates_loss_xy_tf = []
    for transfer in transfer_candidates:
        last_loss_xy_tf_num = None
        for iter_num in range(0, max_iteration):
            out_of_map, loss, map_xy, transfer = cal_new_transfer_and_last_loss_xy(
                transfer, match_seq, mag_map, block_size, step
            )
            if not out_of_map:
                if break_advanced and loss <= target_loss:
                    print("break advanced in the search!")
                    return transfer
                last_loss_xy_tf_num = [loss, map_xy, transfer, iter_num]
            else:
                break

        if last_loss_xy_tf_num is not None:
            candidates_loss_xy_tf.append(last_loss_xy_tf_num)
    # 3.选出候选集中Loss最小的项，返回其transfer；
    #     若无候选项，则表示小范围寻找失败，返回original_transfer
    transfer = None
    min_loss = None
    print("candidates loss:")
    for c in candidates_loss_xy_tf:
        print(c[0])
        if c[0] < target_loss:
            if min_loss is None or c[0] < min_loss:
                min_loss = c[0]
                transfer = c[2]

    if transfer is None:
        print("区域遍历失败，无法找到匹配轨迹！选择相信PDR和之前的transfer")
        return original_transfer
    print("区域遍历成功，找到匹配轨迹！")
    return transfer


# 均值移除
def remove_mean(magSerial):
    return magSerial - np.mean(magSerial)
