import math
import mag_mapping_tools as MMT
import numpy as np
import my_test.test_tools as TEST
import paint_tools as PT
import os
import time

# -----------地图系统参数------------------
MOVE_X = 10.  # iLocator真值坐标平移参数（m）
MOVE_Y = 15.
MAP_SIZE_X = 70.  # 地图坐标系大小 0-MAP_SIZE_X ，0-MAP_SIZE_Y（m）
MAP_SIZE_Y = 28.
BLOCK_SIZE = 0.3  # 地图块大小，（m）
EMD_FILTER_LEVEL = 3  # 低通滤波的程度，值越大滤波越强。整型，无单位。
BUFFER_DIS = 10  # 缓冲池大小（m）
DOWN_SIP_DIS = BLOCK_SIZE  # 下采样粒度（m），应为块大小的整数倍？（下采样越小则相同长度序列的匹配点越多，匹配难度越大！）
# --------迭代搜索参数----------------------
SLIDE_STEP = 2  # 滑动窗口步长
SLIDE_BLOCK_SIZE = DOWN_SIP_DIS  # 滑动窗口最小粒度（m），最小应为下采样粒度！
MAX_ITERATION = 80  # 高斯牛顿最大迭代次数
TARGET_MEAN_LOSS = 50  # 目标损失
STEP = 1 / 50  # 迭代步长，牛顿高斯迭代是局部最优，步长要小
UPPER_LIMIT_OF_GAUSSNEWTEON = 500 * STEP * (MAX_ITERATION - 1)  # 当前参数下高斯牛顿迭代MAX_ITERATION的能降低的loss上限
# ---------其他参数----------------------------
PDR_IMU_ALIGN_SIZE = 10  # 1个PDR坐标对应的imu\iLocator数据个数，iLocator与imu已对齐
TRANSFERS_PRODUCE_CONFIG = [[0.2, 0.2, math.radians(1.2)],  # 枚举transfers的参数，[0 ] = [△x, △y(米), △angle(弧度)]
                            [8, 8, 10]]  # [1] = [枚举的正负个数]
ORIGINAL_START_TRANSFER = [0., 0., math.radians(0.)]  # 初始Transfer[△x, △y(米), △angle(弧度)]：先绕原坐标原点逆时针旋转，然后再平移
# PDR_IMU_START = 20  # PDR舍弃了所使用的IMU数据开头的一定数量的帧数
# ---------数据文件路径---------------------------
# PATH_PDR_RAW = [
#     "./data/InfCenter server room/position_test/5/IMU-812-5-277.2496012617084 Pixel 6_sync.csv.npy",
#     "./data/InfCenter server room/position_test/5/IMU-812-5-277.2496012617084 Pixel 6_sync.csv"]
#
# PATH_PDR_RAW = [
#     "./data/InfCenter server room/position_test/6/IMU-812-6-269.09426660025395 Pixel 6_sync.csv.npy",
#     "./data/InfCenter server room/position_test/6/IMU-812-6-269.09426660025395 Pixel 6_sync.csv"]
#
# PATH_PDR_RAW = [
#     "./data/InfCenter server room/position_test/7/IMU-812-7-195.4948665194862 Pixel 6_sync.csv.npy",
#     "./data/InfCenter server room/position_test/7/IMU-812-7-195.4948665194862 Pixel 6_sync.csv"]
#
# PATH_PDR_RAW = [
#     "./data/InfCenter server room/position_test/8/IMU-812-8-193.38120983931242 Pixel 6_sync.csv.npy",
#     "./data/InfCenter server room/position_test/8/IMU-812-8-193.38120983931242 Pixel 6_sync.csv"]

# PATH_PDR_RAW = [
#     "./data/InfCenter server room/position_test/9/IMU-812-9-189.79622112889115 Pixel 6_sync.csv.npy",
#     "./data/InfCenter server room/position_test/9/IMU-812-9-189.79622112889115 Pixel 6_sync.csv"]

# 地磁指纹库文件，[0]为mv.csv，[1]为mh.csv
# PATH_MAG_MAP = [
#     "./data/InfCenter server room/mag_map/map_F1_2_3_4_B_0.3_deleted/mv_qiu_2d.csv",
#     "./data/InfCenter server room/mag_map/map_F1_2_3_4_B_0.3_deleted/mh_qiu_2d.csv"
# ]

PATH_PDR_RAW = ['./data/XingHu hall 8F test/position_test/6/IMU-88-6-194.9837361431375 Pixel 6_sync.csv.npy', './data/XingHu hall 8F test/position_test/6/IMU-88-6-194.9837361431375 Pixel 6_sync.csv']
PATH_MAG_MAP = ['./data/XingHu hall 8F test/mag_map/map_F1_2_B_0.3_full/mv_qiu_2d.csv', './data/XingHu hall 8F test/mag_map/map_F1_2_B_0.3_full/mh_qiu_2d.csv']


def main():
    result_dir_path = os.path.dirname(PATH_PDR_RAW[0]) + '/result'
    if not os.path.exists(result_dir_path):
        os.mkdir(result_dir_path)
    result_msg_file = open(result_dir_path + '/inf.txt', "w", encoding='GBK')
    paint_map_size = [0, MAP_SIZE_X * 1.0, 0, MAP_SIZE_Y * 1.0]

    print("MOVE_X = {0}\nMOVE_Y = {1}\nMAP_SIZE_X = {2}\nMAP_SIZE_Y = {3}\nBLOCK_SIZE = {4}\nEMD_FILTER_LEVEL = {5}\n"
          "BUFFER_DIS = {6}\nDOWN_SIP_DIS = {7}\nSLIDE_STEP = {8}\nSLIDE_BLOCK_SIZE = {9}\nMAX_ITERATION = {10}\n"
          "TARGET_MEAN_LOSS = {11}\nSTEP = {12}\nUPPER_LIMIT_OF_GAUSSNEWTEON = {13}\nPDR_IMU_ALIGN_SIZE = {14}\n"
          "TRANSFERS_PRODUCE_CONFIG = {15}\nORIGINAL_START_TRANSFER = {16}\n\n"
          "PATH_PDR_GT_IMU = {17}\nPATH_MAG_MAP = {18}\n\n".format(
        MOVE_X, MOVE_Y, MAP_SIZE_X, MAP_SIZE_Y, BLOCK_SIZE, EMD_FILTER_LEVEL,
        BUFFER_DIS, DOWN_SIP_DIS, SLIDE_STEP, SLIDE_BLOCK_SIZE, MAX_ITERATION,
        TARGET_MEAN_LOSS, STEP, UPPER_LIMIT_OF_GAUSSNEWTEON, PDR_IMU_ALIGN_SIZE,
        TRANSFERS_PRODUCE_CONFIG, ORIGINAL_START_TRANSFER,
        PATH_PDR_RAW, PATH_MAG_MAP
    ), file=result_msg_file)
    print("MOVE_X = {0}\nMOVE_Y = {1}\nMAP_SIZE_X = {2}\nMAP_SIZE_Y = {3}\nBLOCK_SIZE = {4}\nEMD_FILTER_LEVEL = {5}\n"
          "BUFFER_DIS = {6}\nDOWN_SIP_DIS = {7}\nSLIDE_STEP = {8}\nSLIDE_BLOCK_SIZE = {9}\nMAX_ITERATION = {10}\n"
          "TARGET_MEAN_LOSS = {11}\nSTEP = {12}\nUPPER_LIMIT_OF_GAUSSNEWTEON = {13}\nPDR_IMU_ALIGN_SIZE = {14}\n"
          "TRANSFERS_PRODUCE_CONFIG = {15}\nORIGINAL_START_TRANSFER = {16}\n\n"
          "PATH_PDR_GT_IMU = {17}\nPATH_MAG_MAP = {18}\n\n".format(
        MOVE_X, MOVE_Y, MAP_SIZE_X, MAP_SIZE_Y, BLOCK_SIZE, EMD_FILTER_LEVEL,
        BUFFER_DIS, DOWN_SIP_DIS, SLIDE_STEP, SLIDE_BLOCK_SIZE, MAX_ITERATION,
        TARGET_MEAN_LOSS, STEP, UPPER_LIMIT_OF_GAUSSNEWTEON, PDR_IMU_ALIGN_SIZE,
        TRANSFERS_PRODUCE_CONFIG, ORIGINAL_START_TRANSFER,
        PATH_PDR_RAW, PATH_MAG_MAP
    ))

    # 全流程
    # 1.建库
    # 读取提前建库的文件，并合并生成原地磁指纹地图mag_map
    mag_map = MMT.rebuild_map_from_mvh_files(PATH_MAG_MAP)
    if mag_map is None:
        print("Mag map rebuild failed!")
        print("Mag map rebuild failed!", file=result_msg_file)
        return
    # PT.paint_heat_map(mag_map, save_dir=result_dir_path + '/')

    # 2、缓冲池给匹配段（内置稀疏采样），此阶段的data与上阶段无关
    pdr_xy = np.load(PATH_PDR_RAW[0])[:, 0:2]
    data_all = MMT.get_data_from_csv(PATH_PDR_RAW[1])
    gt_xy = data_all[:, np.shape(data_all)[1] - 5:np.shape(data_all)[1] - 3]
    # 将iLocator_xy\pdr_xy的坐标平移到MagMap中
    MMT.change_axis(gt_xy, MOVE_X, MOVE_Y)
    MMT.change_axis(pdr_xy, MOVE_X, MOVE_Y)
    PT.paint_xy_list([gt_xy, pdr_xy], ['GT', 'PDR'], paint_map_size, ' ')
    data_mag = data_all[:, 21:24]
    data_quat = data_all[:, 7:11]

    match_seq_list, slide_number_list = MMT.samples_buffer_with_pdr_and_slidewindow(
        BUFFER_DIS, DOWN_SIP_DIS,
        data_quat, data_mag, pdr_xy,
        do_filter=True,
        lowpass_filter_level=EMD_FILTER_LEVEL,
        pdr_imu_align_size=PDR_IMU_ALIGN_SIZE,
        slide_step=SLIDE_STEP,
        slide_block_size=SLIDE_BLOCK_SIZE
    )  # match_seq_list：[?][?][x,y, mv, mh, PDRindex] (多条匹配序列)

    seq_num = len(match_seq_list)
    print("Match seq number:", seq_num, file=result_msg_file)
    print("Match seq number:", seq_num)

    if match_seq_list is None:
        print("Get match seq list failed!")
        print("Get match seq list failed!", file=result_msg_file)
        return

    # 3、迭代匹配段
    #  迭代结束情况：A：迭代out_of_map返回True；B：迭代次数超出阈值但last_loss仍未达标；C：迭代last_loss小于阈值。AB表示匹配失败
    #   3.1 给初始transfer
    transfer = ORIGINAL_START_TRANSFER

    #    3.2 基于初始匹配进行迭代
    start_time = time.time()

    map_xy_list = []
    for i in range(0, len(match_seq_list)):
        print("\nMatch Seq {0}/{1} :".format(i, seq_num))
        print("\nMatch Seq {0}/{1} :".format(i, seq_num), file=result_msg_file)
        match_seq = np.array(match_seq_list[i])  # 待匹配序列match_seq[N][x,y, mv, mh, PDRindex]
        start_transfer = transfer.copy()  # NOTE: Use copy() if just pointer copy caused unexpect data changed
        print("\tStart transfer:[{0:.5}, {1:.5}, {2:.5}°]"
              .format(start_transfer[0], start_transfer[1], math.degrees(start_transfer[2])))
        print("\tStart transfer:[{0:.5}, {1:.5}, {2:.5}°]"
              .format(start_transfer[0], start_transfer[1], math.degrees(start_transfer[2])), file=result_msg_file)
        # 1.核心循环搜索代码
        transfer, map_xy = MMT.produce_transfer_candidates_and_search(start_transfer, TRANSFERS_PRODUCE_CONFIG,
                                                                      match_seq, mag_map,
                                                                      BLOCK_SIZE, STEP, MAX_ITERATION,
                                                                      TARGET_MEAN_LOSS,
                                                                      UPPER_LIMIT_OF_GAUSSNEWTEON,
                                                                      MMT.SearchPattern.BREAKE_ADVANCED_AND_USE_SECOND_LOSS_WHEN_FAILED)
        # 修改每个滑动窗口的实际生效坐标数量
        map_xy = map_xy[0: slide_number_list[i]]
        match_seq = match_seq[0: slide_number_list[i]]
        map_xy_list.append(map_xy)

        # 2.如果找到新的transfer，则计算指纹库的磁场特征
        if not np.array_equal(transfer, start_transfer):
            print("\tFound new transfer:[{0:.5}, {1:.5}, {2:.5}°]"
                  .format(transfer[0], transfer[1], math.degrees(transfer[2])))
            print("\tFound new transfer:[{0:.5}, {1:.5}, {2:.5}°]"
                  .format(transfer[0], transfer[1], math.degrees(transfer[2])), file=result_msg_file)

            temp_map_xy = MMT.transfer_axis_of_xy_seq(match_seq[:, 0:2], transfer)
            mag_map_mvh = []
            mag_map_grads = []
            for xy in temp_map_xy:
                map_mvh, grad = MMT.get_linear_map_mvh_with_grad_2(mag_map, xy[0], xy[1], BLOCK_SIZE)
                # 此时xy取到的grad必为有效值，不需要判断
                mag_map_mvh.append(map_mvh)
                mag_map_grads.append(grad)
            mag_map_mvh = np.array(mag_map_mvh)
            # PT.paint_signal(mag_map_mvh[:, 0], "Map mv Seq {0}".format(i))
            # PT.paint_signal(mag_map_mvh[:, 1], "Map mh Seq {0}".format(i))
            std_deviation_mv, std_deviation_mh, std_deviation_all = MMT.cal_std_deviation_mag_vh(mag_map_mvh)  # 标准差
            unsameness_mv, unsameness_mh, unsameness_all = MMT.cal_unsameness_mag_vh(mag_map_mvh)  # 相邻不相关程度
            grad_level_mv, grad_level_mh, grad_level_all = MMT.cal_grads_level_mag_vh(mag_map_grads)  # 整体梯度水平
            print("\tFeatures of map mag:"
                  "\n\t\t.deviation  mv, mh, all: {0:.4}, {1:.4} = {2:.4}"
                  "\n\t\t.unsameness mv, mh, all: {3:.4}, {4:.4} = {5:.4}"
                  "\n\t\t.grad level mv, mh, all: {6:.4}, {7:.4} = {8:.4}"
                  .format(std_deviation_mv, std_deviation_mh, std_deviation_all,
                          unsameness_mv, unsameness_mh, unsameness_all,
                          grad_level_mv, grad_level_mh, grad_level_all))
            print("\tFeatures of map mag:"
                  "\n\t\t.deviation  mv, mh, all: {0:.4}, {1:.4} = {2:.4}"
                  "\n\t\t.unsameness mv, mh, all: {3:.4}, {4:.4} = {5:.4}"
                  "\n\t\t.grad level mv, mh, all: {6:.4}, {7:.4} = {8:.4}"
                  .format(std_deviation_mv, std_deviation_mh, std_deviation_all,
                          unsameness_mv, unsameness_mh, unsameness_all,
                          grad_level_mv, grad_level_mh, grad_level_all), file=result_msg_file)
            # 现在根据全部已有特征判断当前transfer要不要使用。如果判断不使用，则回退transfer
            if not MMT.trusted_mag_features():
                transfer = start_transfer

        # 3. 计算实测序列中的地磁序列的特征
        mag_vh_arr = match_seq[:, 2:4]
        # PT.paint_signal(mag_vh_arr[:, 0], "Real mv Seq {0}".format(i))
        # PT.paint_signal(mag_vh_arr[:, 1], "Real mh Seq {0}".format(i))
        std_deviation_mv, std_deviation_mh, std_deviation_all = MMT.cal_std_deviation_mag_vh(mag_vh_arr)  # 标准差
        unsameness_mv, unsameness_mh, unsameness_all = MMT.cal_unsameness_mag_vh(mag_vh_arr)  # 相邻不相关程度
        print("\tFeatures of real time mag: "
              "\n\t\t.deviation  mv, mh, all: {0:.4}, {1:.4} = {2:.4}"
              "\n\t\t.unsameness mv, mh, all: {3:.4}, {4:.4} = {5:.4}"
              .format(std_deviation_mv, std_deviation_mh, std_deviation_all,
                      unsameness_mv, unsameness_mh, unsameness_all))
        print("\tFeatures of real time mag: "
              "\n\t\t.deviation  mv, mh, all: {0:.4}, {1:.4} = {2:.4}"
              "\n\t\t.unsameness mv, mh, all: {3:.4}, {4:.4} = {5:.4}"
              .format(std_deviation_mv, std_deviation_mh, std_deviation_all,
                      unsameness_mv, unsameness_mh, unsameness_all), file=result_msg_file)
        # 特征输出完毕，这些print后续可以去掉-----------------------------------------------------------------------------

        # 4.计算该段raw_xy（仅初始对齐的PDR轨迹）\map_xy和真值iLocator_xy的误差距离，并打印输出
        index_list = []
        for p in match_seq:
            index_list.append(p[4])
        index_list = np.array(index_list)
        index_list = index_list[:, np.newaxis]
        # 轨迹与pdr原始下标合并`
        map_xy_with_index = np.concatenate((map_xy, index_list), axis=1)
        raw_xy = MMT.transfer_axis_of_xy_seq(match_seq[:, 0:2], ORIGINAL_START_TRANSFER)
        raw_xy_with_index = np.concatenate((raw_xy, index_list), axis=1)
        # 计算轨迹距离
        distance_of_MagPDR_iLocator_points = TEST.cal_distance_between_GT_and_MagPDR(
            gt_xy, map_xy_with_index, xy_align_size=PDR_IMU_ALIGN_SIZE)
        distance_of_PDR_iLocator_points = TEST.cal_distance_between_GT_and_MagPDR(
            gt_xy, raw_xy_with_index, xy_align_size=PDR_IMU_ALIGN_SIZE)
        mean_distance_between_MagPDR_GT = np.mean(distance_of_MagPDR_iLocator_points[:, 0])
        mean_distance_between_PDR_GT = np.mean(distance_of_PDR_iLocator_points[:, 0])
        improvement = mean_distance_between_PDR_GT - mean_distance_between_MagPDR_GT
        print("\tMean Distance between PDR and GT: %.3f" % mean_distance_between_PDR_GT)
        print("\tMean Distance between MagPDR and GT: %.3f" % mean_distance_between_MagPDR_GT)
        print("\tImprovement: %.3f" % improvement)
        print("\tMean Distance between PDR and GT: %.3f" % mean_distance_between_PDR_GT, file=result_msg_file)
        print("\tMean Distance between MagPDR and GT: %.3f" % mean_distance_between_MagPDR_GT, file=result_msg_file)
        print("\tImprovement: %.3f" % improvement, file=result_msg_file)

    end_time = time.time()
    # -----------4 计算结果参数------------------------------------------------------------------------------------------
    print("\n\n====================MagPDR End =============================================")
    print("Calculate and show the Evaluation results:")
    print("\n\n====================MagPDR End =============================================", file=result_msg_file)
    print("Calculate and show the Evaluation results:", file=result_msg_file)
    # 4.1.png 将计算的分段mag xy合并还原为一整段 final_xy
    final_xy = []
    final_index = []
    for map_xy in map_xy_list:
        for xy in map_xy:
            final_xy.append(xy)
    final_xy = np.array(final_xy)
    # 4.2 还原每个xy对应的原PDR中的下标index
    for i in range(0, len(match_seq_list)):
        for p in match_seq_list[i][0: slide_number_list[i]]:
            final_index.append(p[4])
    # 4.3 将final_xy与final_index合并为MagPDR_xy（合并前要先在final_index的列上增加维度，让其由1维变为N×1的二维数组）
    final_index = np.array(final_index)
    final_index = final_index[:, np.newaxis]
    magPDR_xy = np.concatenate((final_xy, final_index), axis=1)
    # 4.4 利用前面记录的初始变换向量start_transfer，将PDR xy转换至MagMap中，作为未经过较准的对照组
    pdr_xy = MMT.transfer_axis_of_xy_seq(pdr_xy, ORIGINAL_START_TRANSFER)
    # 4.5 计算PDR xy与Ground Truth(iLocator)之间的单点距离
    distance_of_PDR_iLocator_points = TEST.cal_distance_between_GT_and_PDR(
        gt_xy, pdr_xy, xy_align_size=PDR_IMU_ALIGN_SIZE)
    # 4.6 计算MagPDR xy与Ground Truth(iLocator)之间的单点距离
    distance_of_MagPDR_iLocator_points = TEST.cal_distance_between_GT_and_MagPDR(
        gt_xy, magPDR_xy, xy_align_size=PDR_IMU_ALIGN_SIZE)
    # 4.7 计算整段轨迹长度
    traj_length_dis = 0
    for i in range(1, len(pdr_xy)):
        traj_length_dis += math.hypot(pdr_xy[i][0] - pdr_xy[i - 1][0], pdr_xy[i][1] - pdr_xy[i - 1][1])

    np.savetxt(result_dir_path + '/pdr_gt_dis.csv', distance_of_PDR_iLocator_points, delimiter=',')
    np.savetxt(result_dir_path + '/magPdr_gt_dis.csv', distance_of_MagPDR_iLocator_points, delimiter=',')

    # -----------5 输出结果参数------------------------------------------------------------------------------------------
    # 5.1 打印PDR xy与Ground Truth(iLocator)之间的单点距离、平均距离
    mean_distance = np.mean(distance_of_PDR_iLocator_points[:, 0])
    print("\tMean Distance between PDR and GT: ", mean_distance)
    print("\tMean Distance between PDR and GT: ", mean_distance, file=result_msg_file)

    # 5.2 打印MagPDR xy与Ground Truth(iLocator)之间的单点距离、平均距离
    mean_distance = np.mean(distance_of_MagPDR_iLocator_points[:, 0])
    print("\tMean Distance between MagPDR and GT: ", mean_distance)
    print("\tMean Distance between MagPDR and GT: ", mean_distance, file=result_msg_file)

    # 打印magPDR与iLocator距离的一倍σ参数：65%的坐标与真值的距离是1m以内。
    target_percent, sigma_percent = TEST.cal_sigma_level(1, distance_of_MagPDR_iLocator_points)
    print("\tTarget and Sigma percent between MagPDR and GT:", target_percent, sigma_percent)
    print("\tTarget and Sigma percent between MagPDR and GT:", target_percent, sigma_percent, file=result_msg_file)

    # 5.3 对Ground Truth(iLocator)、PDR、MagPDR进行绘图
    PT.paint_xy_list([gt_xy], ["GT by iLocator"], paint_map_size, ' ', save_file=result_dir_path + '/GT by iLocator.png')
    PT.paint_xy_list([pdr_xy], ["PDR"], paint_map_size, ' ', save_file=result_dir_path + '/PDR.png')
    PT.paint_xy_list([final_xy],
                     ['MagPDR'],
                     paint_map_size,
                     "The MagPDR: BlockSize={0}, BufferDis={1}, MaxIteration={2}, Step={3:.8f}, TargetLoss={4}"
                     .format(BLOCK_SIZE, BUFFER_DIS, MAX_ITERATION, STEP, TARGET_MEAN_LOSS),
                     save_file=result_dir_path + '/MagPDR.png')
    PT.paint_xy_list([gt_xy, pdr_xy], ['GT', 'PDR'], paint_map_size, 'Contrast of Lines',
                     save_file=result_dir_path + '/GT PDR.png')
    PT.paint_xy_list([gt_xy, final_xy], ['GT', 'MagPDR'], paint_map_size, "Contrast of Lines",
                     save_file=result_dir_path + '/GT MagPDR.png')
    PT.paint_xy_list([gt_xy, pdr_xy, final_xy], ['GT', 'PDR', 'MagPDR'], paint_map_size, "Contrast of Lines",
                     save_file=result_dir_path + '/GT PDR MagPDR.png')

    result_msg_file.close()

    # 额外参数
    print("Cost time:", end_time - start_time, " second")
    print("Traj length", traj_length_dis, " m")
    print("Cost time/Traj", (end_time - start_time)/traj_length_dis, " s/m")
    return


if __name__ == '__main__':
    main()
