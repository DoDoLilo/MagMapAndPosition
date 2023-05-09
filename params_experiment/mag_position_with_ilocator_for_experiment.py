import math
import time

import mag_mapping_tools as MMT
import numpy as np
import my_test.test_tools as TEST


# 改成用参数调用
#   注释掉计算features的逻辑
def mag_position_with_ilcator(
        MOVE_X, MOVE_Y,
        BLOCK_SIZE,
        EMD_FILTER_LEVEL,
        BUFFER_DIS, DOWN_SIP_DIS, SLIDE_STEP, SLIDE_BLOCK_SIZE,
        MAX_ITERATION, TARGET_MEAN_LOSS, STEP, UPPER_LIMIT_OF_GAUSSNEWTEON,
        PDR_IMU_ALIGN_SIZE,
        TRANSFERS_PRODUCE_CONFIG,
        ORIGINAL_START_TRANSFER,
        PATH_PDR_RAW, PATH_MAG_MAP
):
    # 全流程
    # 1.建库
    # 读取提前建库的文件，并合并生成原地磁指纹地图mag_map
    mag_map = MMT.rebuild_map_from_mvh_files(PATH_MAG_MAP)
    if mag_map is None:
        print("Mag map rebuild failed!")
        return None

    # 2、缓冲池给匹配段（内置稀疏采样），此阶段的data与上阶段无关
    pdr_xy = np.load(PATH_PDR_RAW[0])[:, 0:2]
    data_all = MMT.get_data_from_csv(PATH_PDR_RAW[1])
    gt_xy = data_all[:, np.shape(data_all)[1] - 5:np.shape(data_all)[1] - 3]
    # 将iLocator_xy\pdr_xy的坐标平移到MagMap中
    MMT.change_axis(gt_xy, MOVE_X, MOVE_Y)
    MMT.change_axis(pdr_xy, MOVE_X, MOVE_Y)
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
    # print("Match seq number:", seq_num)

    if match_seq_list is None:
        print("Get match seq list failed!")
        return None

    # 3、迭代匹配段
    #  迭代结束情况：A：迭代out_of_map返回True；B：迭代次数超出阈值但last_loss仍未达标；C：迭代last_loss小于阈值。AB表示匹配失败
    #   3.1 给初始transfer
    transfer = ORIGINAL_START_TRANSFER

    #    3.2 基于初始匹配进行迭代
    start_time = time.time()
    map_xy_list = []
    for i in range(0, len(match_seq_list)):
        match_seq = np.array(match_seq_list[i])  # 待匹配序列match_seq[N][x,y, mv, mh, PDRindex]
        start_transfer = transfer.copy()  # NOTE: Use copy() if just pointer copy caused unexpect data changed
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

    end_time = time.time()

    # -----------4 计算结果参数------------------------------------------------------------------------------------------
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
    traj_dis = 0
    for i in range(1, len(pdr_xy)):
        traj_dis += math.hypot(pdr_xy[i][0] - pdr_xy[i - 1][0], pdr_xy[i][1] - pdr_xy[i - 1][1])

    # -----------5 输出结果参数------------------------------------------------------------------------------------------
    # 5.1.png 打印PDR xy与Ground Truth(iLocator)之间的单点距离、平均距离
    mean_distance_of_pdr_gt = np.mean(distance_of_PDR_iLocator_points[:, 0])

    # 5.2 打印MagPDR xy与Ground Truth(iLocator)之间的单点距离、平均距离
    mean_distance_of_mag_gt = np.mean(distance_of_MagPDR_iLocator_points[:, 0])

    # 打印magPDR与iLocator距离的一倍σ参数：65%的坐标与真值的距离是1m以内。
    less_1m_percent, sigma_percent = TEST.cal_sigma_level(1, distance_of_MagPDR_iLocator_points)

    # 额外参数
    cost_time = end_time - start_time
    cost_time_per_meter = traj_dis / cost_time

    return [mean_distance_of_pdr_gt, mean_distance_of_mag_gt, less_1m_percent, sigma_percent, cost_time, cost_time_per_meter, traj_dis]


