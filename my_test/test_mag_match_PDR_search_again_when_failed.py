import math
import mag_mapping_tools as MMT
import numpy as np
import my_test.test_tools as TEST

# iLocator真值坐标平移参数
MOVE_X = 7
MOVE_Y = 8
# 地图坐标系大小 0-MAP_SIZE_X ，0-MAP_SIZE_Y
MAP_SIZE_X = 8.
MAP_SIZE_Y = 13.
# 地图地磁块大小
BLOCK_SIZE = 0.25
# 低通滤波的程度，值越大滤波越强。整型，无单位。
EMD_FILTER_LEVEL = 3
# ------------------------
# 缓冲池大小，单位（m）
BUFFER_DIS = 5
# 下采样粒度，应为块大小的整数倍？（下采样越小，匹配难度越大！）
DOWN_SIP_DIS = BLOCK_SIZE
# 高斯牛顿最大迭代次数
MAX_ITERATION = 90
# 目标损失
TARGET_LOSS = BUFFER_DIS / BLOCK_SIZE * 15
print("TARGET_LOSS:", TARGET_LOSS, '\n')
# 迭代步长，牛顿高斯迭代是局部最优，步长要小
STEP = 1 / 70
# 原始数据采样频率 , PDR坐标输出频率
SAMPLE_FREQUENCY = 200
PDR_XY_FREQUENCY = 20
# 首次迭代固定区域遍历数组，默认起点在某一固定区域，transfer=[△x,△y,△angle]，
# Transfer[△x, △y(米), △angle(弧度)]：先绕原坐标原点逆时针旋转，然后再平移
ORIGINAL_START_TRANSFER = [6.7, 1.7, math.radians(-98.)]
START_CONFIG = [[0.25, 0.25, math.radians(1.)], [3, 3, 3]]
START_TRANSFERS = MMT.produce_transfer_candidates_ascending(ORIGINAL_START_TRANSFER, START_CONFIG)
PATH_PDR_RAW = ["../data/data_test/pdr/IMU-10-2-183.5307793202117 Pixel 6.csv.npy",
                "../data/data_test/data_to_building_map/IMU-10-2-183.5307793202117 Pixel 6_sync.csv"]


def main():
    # 全流程
    # 1、建库
    # 读取提前建库的文件，并合并生成原地磁指纹地图mag_map
    mag_map_mv = np.array(np.loadtxt('../data/data_test/mag_map/mag_map_mv.csv', delimiter=','))
    mag_map_mh = np.array(np.loadtxt('../data/data_test/mag_map/mag_map_mh.csv', delimiter=','))
    mag_map = []
    for i in range(0, len(mag_map_mv)):
        temp = []
        for j in range(0, len(mag_map_mv[0])):
            temp.append([mag_map_mv[i][j], mag_map_mh[i][j]])
        mag_map.append(temp)
    mag_map = np.array(mag_map)
    # MMT.paint_heat_map(mag_map, show_mv=False)

    # 2、缓冲池给匹配段（内置稀疏采样），此阶段的data与上阶段无关
    pdr_xy = np.load(PATH_PDR_RAW[0])[:, 0:2]
    data_all = MMT.get_data_from_csv(PATH_PDR_RAW[1])
    iLocator_xy = data_all[:, np.shape(data_all)[1] - 5:np.shape(data_all)[1] - 3]
    raw_mag = data_all[:, 21:24]
    raw_ori = data_all[:, 18:21]
    # match_seq_list=[?][?][x,y, mv, mh, PDRindex] (多条匹配序列)
    match_seq_list = MMT.samples_buffer_PDR(BUFFER_DIS, DOWN_SIP_DIS, raw_ori, raw_mag, pdr_xy,
                                            do_filter=True, lowpass_filter_level=EMD_FILTER_LEVEL,
                                            pdr_frequency=PDR_XY_FREQUENCY, sampling_frequency=SAMPLE_FREQUENCY)

    # 3、根据匹配段进行迭代，3种迭代结束情况：
    #  A：迭代out_of_map返回True；B：迭代次数超出阈值但last_loss仍未达标；C：迭代last_loss小于阈值
    # 情况AB表示匹配失败
    #   3.1第一次匹配进行起点周围区域遍历，遍历后选出残差最小的
    first_match = np.array(match_seq_list[0])
    print("*Match Seq 0 : \n\tfirst deep search seq.")
    transfer = MMT.produce_transfer_candidates_search_again(ORIGINAL_START_TRANSFER, START_CONFIG,
                                                            first_match, mag_map, BLOCK_SIZE, STEP, MAX_ITERATION,
                                                            TARGET_LOSS, break_advanced=False)
    out_of_map, loss, map_xy, not_used_transfer = MMT.cal_new_transfer_and_last_loss_xy(
        transfer, first_match, mag_map, BLOCK_SIZE, STEP
    )
    if out_of_map or loss > TARGET_LOSS:
        print("\t初始Transfer遍历查找失败!")
        return

    print("\tThe Start Track(Loss={0}, Target Loss={2},Transfer={1}):"
          .format(loss, [transfer[0], transfer[1], math.degrees(transfer[2])], TARGET_LOSS))
    start_transfer = transfer.copy()

    #    3.2后续轨迹基于初始匹配进行迭代------------------------------------------------------------------------------------------
    map_xy_list = [map_xy]

    for i in range(1, len(match_seq_list)):
        # 获取实测待匹配序列match_seq[N][x,y, mv, mh, PDRindex]
        match_seq = np.array(match_seq_list[i])
        iter_num = 0
        loss_list = []
        start_transfer = transfer.copy()  # NOTE: Use copy() if just pointer copy caused unexpect data changed

        # 计算实测序列中的地磁序列的特征程度
        mag_vh_arr = match_seq[:, 2:4]
        std_deviation_mv, std_deviation_mh, std_deviation_all = MMT.cal_std_deviation_mag_vh(mag_vh_arr)  # 标准差
        unsameness_mv, unsameness_mh, unsameness_all = MMT.cal_unsameness_mag_vh(mag_vh_arr)  # 相邻不相关程度

        print("\nMatch Seq {0} :".format(i))
        print("\t.real time mag features: Standard Deviation mv, mh, all: {0:.4}, {1:.4} = {2:.4} | "
              "Unsameness mv, mh, all:{3:.4}, {4:.4} = {5:.4}"
              .format(std_deviation_mv, std_deviation_mh, std_deviation_all,
                      unsameness_mv, unsameness_mh, unsameness_all))
        print("\t.start_transfer:", [transfer[0], transfer[1], math.degrees(transfer[2])])

        while True:
            iter_num += 1
            out_of_map, loss, map_xy, transfer = MMT.cal_new_transfer_and_last_loss_xy(
                transfer, match_seq, mag_map, BLOCK_SIZE, STEP
            )
            if not out_of_map:
                loss_list.append(loss)

            if out_of_map or iter_num > MAX_ITERATION:
                print("\tFailed in iteration", iter_num, ", loss list:", loss_list)
                # 失败了怎么办？
                # 1.在迭代前备份的start_transfer基础上进行范围搜索，范围从近到远，匹配成功则提前结束。
                print("\tSearch more in the start_transfer ... ...")
                transfer = MMT.produce_transfer_candidates_search_again(start_transfer, START_CONFIG,
                                                                        match_seq, mag_map,
                                                                        BLOCK_SIZE, STEP, MAX_ITERATION,
                                                                        TARGET_LOSS, break_advanced=True)
                # 2.若仍失败，则相信PDR和last_transfer，接着输出结果
                # NOTE:失败了还要还原之前的transfer，因为迭代过程中transfer一直在变化！
                map_xy_list.append(MMT.transfer_axis_list(match_seq[:, 0:2], transfer))
                break

            if loss <= TARGET_LOSS:
                print("\tSucceed in iteration", iter_num, ", transfer changed to: ",
                      [transfer[0], transfer[1], math.degrees(transfer[2])])
                # 成功了怎么办？加到结果中，提前结束迭代
                map_xy_list.append(map_xy)
                break

    # -----------4 计算结果参数------------------------------------------------------------------------------------------
    print("\n\n====================MagPDR End =============================================")
    print("Calculate and show the Evaluation results:")
    # 4.1 将计算的分段mag xy合并还原为一整段 final_xy
    final_xy = []
    final_index = []
    for map_xy in map_xy_list:
        for xy in map_xy:
            final_xy.append(xy)
    final_xy = np.array(final_xy)
    # 4.2 还原每个xy对应的原PDR中的下标index
    for match_seq in match_seq_list:
        for p in match_seq:
            final_index.append(p[4])
    # 4.3 将final_xy与final_index合并为MagPDR_xy（合并前要先在final_index的列上增加维度，让其由1维变为N×1的二维数组）
    final_index = np.array(final_index)
    final_index = final_index[:, np.newaxis]
    MagPDR_xy = np.concatenate((final_xy, final_index), axis=1)
    # 4.4 将iLocator_xy的坐标转换到MagMap中，作为Ground Truth
    MMT.change_axis(iLocator_xy, MOVE_X, MOVE_Y)
    # 4.5 利用前面记录的初始变换向量start_transfer，将PDR xy转换至MagMap中，作为未经过较准的对照组
    pdr_xy = MMT.transfer_axis_list(pdr_xy, start_transfer)
    # 4.6 计算PDR xy与Ground Truth(iLocator)之间的单点距离
    distance_of_PDR_iLocator_points = TEST.cal_distance_between_iLocator_and_PDR(
        iLocator_xy, pdr_xy, SAMPLE_FREQUENCY, PDR_XY_FREQUENCY)
    # 4.7 计算MagPDR xy与Ground Truth(iLocator)之间的单点距离
    distance_of_MagPDR_iLocator_points = TEST.cal_distance_between_iLocator_and_MagPDR(
        iLocator_xy, MagPDR_xy, SAMPLE_FREQUENCY, PDR_XY_FREQUENCY)

    # -----------5 输出结果参数------------------------------------------------------------------------------------------
    # 5.1 打印PDR xy与Ground Truth(iLocator)之间的单点距离、平均距离
    # print("distance_of_PDR_iLocator_points:\n", distance_of_PDR_iLocator_points)
    mean_distance = np.mean(distance_of_PDR_iLocator_points[:, 0])
    print("\tMean Distance between PDR and GT: ", mean_distance)
    # 5.2 打印MagPDR xy与Ground Truth(iLocator)之间的单点距离、平均距离
    # print("distance_of_MagPDR_iLocator_points:\n", distance_of_MagPDR_iLocator_points)
    mean_distance = np.mean(distance_of_MagPDR_iLocator_points[:, 0])
    print("\tMean Distance between MagPDR and GT: ", mean_distance)
    # 5.3 对Ground Truth(iLocator)、PDR、MagPDR进行绘图
    MMT.paint_xy(iLocator_xy, "The Ground Truth by iLocator", [0, MAP_SIZE_X * 1.0, 0, MAP_SIZE_Y * 1.0])
    MMT.paint_xy(pdr_xy, "The PDR", [0, MAP_SIZE_X * 1.0, 0, MAP_SIZE_Y * 1.0])
    MMT.paint_xy(final_xy, "The MagPDR: BlockSize={0}, BufferDis={1}, MaxIteration={2}, Step={3:.8f}, TargetLoss={4}"
                 .format(BLOCK_SIZE, BUFFER_DIS, MAX_ITERATION, STEP, TARGET_LOSS),
                 [0, MAP_SIZE_X * 1.0, 0, MAP_SIZE_Y * 1.0])
    return


if __name__ == '__main__':
    main()
