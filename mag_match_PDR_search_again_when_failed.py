import math
import mag_mapping_tools as MMT
import numpy as np

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
TARGET_LOSS = BUFFER_DIS / BLOCK_SIZE * 20
print("TARGET_LOSS:", TARGET_LOSS)
# 迭代步长，牛顿高斯迭代是局部最优，步长要小
STEP = 1 / 70
# 原始数据采样频率 , PDR坐标输出频率
SAMPLE_FREQUENCY = 200
PDR_XY_FREQUENCY = 20
# 首次迭代固定区域遍历数组，默认起点在某一固定区域，transfer=[△x,△y,△angle]，
# Transfer[△x, △y(米), △angle(弧度)]：先绕原坐标原点逆时针旋转，然后再平移
ORIGINAL_START_TRANSFER = [6.7, 1.75, math.radians(-100.)]
START_CONFIG = [[0.25, 0.25, math.radians(1.2)], [3, 3, 3]]
START_TRANSFERS = MMT.produce_transfer_candidates(ORIGINAL_START_TRANSFER, START_CONFIG)
PATH_PDR_RAW = ["data/data_test/pdr/IMU-10-5-170.2125898151382 Pixel 6.csv.npy",
                "data/data_test/data_to_building_map/IMU-10-5-170.2125898151382 Pixel 6_sync.csv"]


def main():
    # 全流程
    # 1、建库
    # 读取提前建库的文件，并合并生成原地磁指纹地图mag_map
    mag_map_mv = np.array(np.loadtxt('data/data_test/mag_map/mag_map_mv.csv', delimiter=','))
    mag_map_mh = np.array(np.loadtxt('data/data_test/mag_map/mag_map_mh.csv', delimiter=','))
    mag_map = []
    for i in range(0, len(mag_map_mv)):
        temp = []
        for j in range(0, len(mag_map_mv[0])):
            temp.append([mag_map_mv[i][j], mag_map_mh[i][j]])
        mag_map.append(temp)
    mag_map = np.array(mag_map)
    MMT.paint_heat_map(mag_map, show_mv=False)

    # 2、缓冲池给匹配段（内置稀疏采样），此阶段的data与上阶段无关
    # TODO 如果3.1失败则重新给出 BUFFER_DIS\DOWN_SIP_DIS\TARGET_LOSS重新开始？
    pdr_xy = np.load(PATH_PDR_RAW[0])[:, 0:2]
    data_all = MMT.get_data_from_csv(PATH_PDR_RAW[1])
    raw_xy = data_all[:, np.shape(data_all)[1] - 5:np.shape(data_all)[1] - 3]
    raw_mag = data_all[:, 21:24]
    raw_ori = data_all[:, 18:21]
    # 并不在此时修改pdr_xy坐标，match_seq_list=多条匹配序列[?][?][x,y, mv, mh]
    match_seq_list = MMT.samples_buffer_PDR(BUFFER_DIS, DOWN_SIP_DIS, raw_ori, raw_mag, pdr_xy,
                                            do_filter=True, lowpass_filter_level=EMD_FILTER_LEVEL,
                                            pdr_frequency=PDR_XY_FREQUENCY, sampling_frequency=SAMPLE_FREQUENCY)

    # 3、手动给出初始transfer_0，注意单条
    # 根据匹配段进行迭代，3种迭代结束情况：
    #  A：迭代out_of_map返回True；B：迭代次数超出阈值但last_loss仍未达标；C：迭代last_loss小于阈值
    # 情况AB表示匹配失败，论文未说匹配失败的解决方法
    #   3.1第一次匹配进行起点周围区域遍历，遍历后选出残差最小的
    first_match = np.array(match_seq_list[0])
    transfer = MMT.produce_transfer_candidates_search_again(ORIGINAL_START_TRANSFER, START_CONFIG,
                                                            first_match, mag_map, BLOCK_SIZE, STEP, MAX_ITERATION,
                                                            TARGET_LOSS)
    out_of_map, loss, map_xy, not_used_transfer = MMT.cal_new_transfer_and_last_loss_xy(
        transfer, first_match, mag_map, BLOCK_SIZE, STEP
    )
    if loss > TARGET_LOSS:
        print("Start matching failed!")
        return

    print("The Start Track(Loss={0}, Target Loss={2},Transfer={1}):"
          .format(loss, [transfer[0], transfer[1], math.degrees(transfer[2])], TARGET_LOSS))
    #    3.2后续轨迹基于初始匹配进行迭代------------------------------------------------------------------------------------------
    map_xy_list = [map_xy]
    for i in range(1, len(match_seq_list)):
        match_seq = np.array(match_seq_list[i])
        print("\nMatch Seq:{0}   --------------------------".format(i))
        iter_num = 0
        loss_list = []
        # *NOTE: Use copy() if just pointer copy caused data changed in the last_transfer
        last_transfer = transfer.copy()
        print("last_transfer:", [transfer[0], transfer[1], math.degrees(transfer[2])])

        while True:
            iter_num += 1
            out_of_map, loss, map_xy, transfer = MMT.cal_new_transfer_and_last_loss_xy(
                transfer, match_seq, mag_map, BLOCK_SIZE, STEP
            )
            if not out_of_map:
                loss_list.append(loss)

            if out_of_map or iter_num > MAX_ITERATION:
                print("Match Seq:", i, "iteration ", iter_num, ", Failed!")
                # 失败了怎么办？
                # 选择：相信PDR和之前的transfer
                # NOTE:失败了还要还原之前的transfer，因为迭代过程中transfer一直在变化！！！
                # 如果不相信PDR，则和第一条序列一样，给当前序列也进行小范围遍历
                print("中间轨迹匹配失败！选择：以之前的transfer重新小范围遍历",
                      [last_transfer[0], last_transfer[1], math.degrees(last_transfer[2])])
                print("Loss list:", loss_list)
                transfer = MMT.produce_transfer_candidates_search_again(last_transfer, START_CONFIG,
                                                                        match_seq, mag_map,
                                                                        BLOCK_SIZE, STEP, MAX_ITERATION,
                                                                        TARGET_LOSS)
                # 如果区域遍历寻找失败，transfer = last_transfer
                map_xy_list.append(MMT.transfer_axis_list(match_seq[:, 0:2], transfer))
                break

            if loss <= TARGET_LOSS:
                print("Match Seq:", i, "iteration ", iter_num, ", Succeed!")
                print("Transfer changed: ", [transfer[0], transfer[1], math.degrees(transfer[2])])
                # 成功了怎么办？加到结果中，提前结束迭代
                map_xy_list.append(map_xy)
                break

    final_xy = []
    for map_xy in map_xy_list:
        for xy in map_xy:
            final_xy.append(xy)
    final_xy = np.array(final_xy)
    MMT.change_axis(raw_xy, MOVE_X, MOVE_Y)
    MMT.paint_xy(raw_xy, "The ground truth by iLocator", [0, MAP_SIZE_X * 1.0, 0, MAP_SIZE_Y * 1.0])
    MMT.paint_xy(final_xy, "The final xy: BlockSize={0}, BufferDis={1}, MaxIteration={2}, Step={3}, TargetLoss={4}"
                 .format(BLOCK_SIZE, BUFFER_DIS, MAX_ITERATION, STEP, TARGET_LOSS),
                 [0, MAP_SIZE_X * 1.0, 0, MAP_SIZE_Y * 1.0])
    return


if __name__ == '__main__':
    main()
