# TODO 针对论文5.1.1节的BFS范围和磁场序列长度，对已有所有测试轨迹：
#  1、测试不同BFS范围对定位效果的影响；
#  2、不同匹配序列长度对定位效果的影响；Max Mean MeanTime MaxWinTime；
#  3、for循环遍历测试文件，保存测试轨迹名，

import math
import mag_mapping_tools as MMT
import numpy as np
import my_test.test_tools as TEST
import paint_tools as PT
import os
import time

from adjust_pdr_by_markpoints import two_slope_angle_off

# -----------地图系统参数------------------
# 星湖楼
MOVE_X = 10.0
MOVE_Y = 15.0
MAP_SIZE_X = 70.0
MAP_SIZE_Y = 28.0

# 信息中心
# MOVE_X = 5.0
# MOVE_Y = 5.0
# MAP_SIZE_X = 35.0
# MAP_SIZE_Y = 20.0

BLOCK_SIZE = 0.3  # 地图块大小，（m）
EMD_FILTER_LEVEL = 3  # 低通滤波的程度，值越大滤波越强。整型，无单位。
BUFFER_DIS = 8  # 缓冲池大小（m）
DOWN_SIP_DIS = BLOCK_SIZE  # 下采样粒度（m），应为块大小的整数倍？（下采样越小则相同长度序列的匹配点越多，匹配难度越大！）
# --------迭代搜索参数----------------------
SLIDE_STEP = 2  # 滑动窗口步长
SLIDE_BLOCK_SIZE = DOWN_SIP_DIS  # 滑动窗口最小粒度（m），最小应为下采样粒度！
MAX_ITERATION = 80  # 高斯牛顿最大迭代次数
TARGET_MEAN_LOSS = 50  # 目标损失
STEP = 1 / 50  # 迭代步长，牛顿高斯迭代是局部最优，步长要小
UPPER_LIMIT_OF_GAUSSNEWTEON = TARGET_MEAN_LOSS * 10  # 当前参数下高斯牛顿迭代MAX_ITERATION的能降低的loss上限
# ---------其他参数----------------------------
PDR_IMU_ALIGN_SIZE = 10  # 1个PDR坐标对应的imu\iLocator数据个数，iLocator与imu已对齐
TRANSFERS_PRODUCE_CONFIG = [[0.2, 0.2, math.radians(1.2)],  # 枚举transfers的参数，[0 ] = [△x, △y(米), △angle(弧度)]
                            [8, 8, 10]]  # [1] = [枚举的正负个数]
ORIGINAL_START_TRANSFER = [0., 0., math.radians(0.)]  # 初始Transfer[△x, △y(米), △angle(弧度)]：先绕原坐标原点逆时针旋转，然后再平移

# PATH_PDR_RAW_s = [
# [
#     "../data/InfCenter server room/position_test/5/IMU-812-5-277.2496012617084 Pixel 6_sync.csv.npy",
#     "../data/InfCenter server room/position_test/5/IMU-812-5-277.2496012617084 Pixel 6_sync.csv"],
# [
#     "../data/InfCenter server room/position_test/6/IMU-812-6-269.09426660025395 Pixel 6_sync.csv.npy",
#     "../data/InfCenter server room/position_test/6/IMU-812-6-269.09426660025395 Pixel 6_sync.csv"],
# [
#     "../data/InfCenter server room/position_test/7/IMU-812-7-195.4948665194862 Pixel 6_sync.csv.npy",
#     "../data/InfCenter server room/position_test/7/IMU-812-7-195.4948665194862 Pixel 6_sync.csv"],
# [
#     "../data/InfCenter server room/position_test/8/IMU-812-8-193.38120983931242 Pixel 6_sync.csv.npy",
#     "../data/InfCenter server room/position_test/8/IMU-812-8-193.38120983931242 Pixel 6_sync.csv"],
# [
#     "../data/InfCenter server room/position_test/9/IMU-812-9-189.79622112889115 Pixel 6_sync.csv.npy",
#     "../data/InfCenter server room/position_test/9/IMU-812-9-189.79622112889115 Pixel 6_sync.csv"]
# ]
#
# # 地磁指纹库文件，[0]为mv.csv，[1]为mh.csv
# PATH_MAG_MAP = [
#     "../data/InfCenter server room/mag_map/map_F1_2_3_4_B_0.3_deleted/mv_qiu_2d.csv",
#     "../data/InfCenter server room/mag_map/map_F1_2_3_4_B_0.3_deleted/mh_qiu_2d.csv"
# ]


PATH_PDR_RAW_s = [
    ['../data/XingHu hall 8F test/position_test/5/IMU-88-5-291.0963959547511 Pixel 6_sync.csv.npy',
     '../data/XingHu hall 8F test/position_test/5/IMU-88-5-291.0963959547511 Pixel 6_sync.csv'],
    ['../data/XingHu hall 8F test/position_test/6/IMU-88-6-194.9837361431375 Pixel 6_sync.csv.npy',
     '../data/XingHu hall 8F test/position_test/6/IMU-88-6-194.9837361431375 Pixel 6_sync.csv'],
    ['../data/XingHu hall 8F test/position_test/7/IMU-88-7-270.6518297687728 Pixel 6_sync.csv.npy',
     '../data/XingHu hall 8F test/position_test/7/IMU-88-7-270.6518297687728 Pixel 6_sync.csv'],
    ['../data/XingHu hall 8F test/position_test/8/IMU-88-8-189.88230883318997 Pixel 6_sync.csv.npy',
     '../data/XingHu hall 8F test/position_test/8/IMU-88-8-189.88230883318997 Pixel 6_sync.csv']
]

PATH_MAG_MAP = ['../data/XingHu hall 8F test/mag_map/map_F1_2_3_4_B_0.3_full/mv_qiu_2d.csv',
                '../data/XingHu hall 8F test/mag_map/map_F1_2_3_4_B_0.3_full/mh_qiu_2d.csv']

BUFFER_DIS_s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,20]

def main():
    traj_kpi_dict = {}
    for PATH_PDR_RAW in PATH_PDR_RAW_s:
        traj_kpi_dict[PATH_PDR_RAW[0]] = []

    for BUFFER_DIS in BUFFER_DIS_s:
        print(BUFFER_DIS)
        all_magxy_gtxy = []  # [N][mx, my, gx, gy]
        max_win_mean_error = 0
        max_seq_time = 0  # 某段窗口的最长耗时
        all_cost_time = 0  # 全部轨迹耗时
        for PATH_PDR_RAW in PATH_PDR_RAW_s:
            print(PATH_PDR_RAW[0])
            traj_time = 0

            result_dir_path = os.path.dirname(PATH_PDR_RAW[0]) + '/paper5_result'
            if not os.path.exists(result_dir_path):
                os.mkdir(result_dir_path)

            # 1.建库
            # 读取提前建库的文件，并合并生成原地磁指纹地图mag_map
            mag_map = MMT.rebuild_map_from_mvh_files(PATH_MAG_MAP)
            if mag_map is None:
                print("Mag map rebuild failed!")
                return

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

            if match_seq_list is None:
                print("Get match seq list failed!")
                return

            # 3、迭代匹配段
            #  迭代结束情况：A：迭代out_of_map返回True；B：迭代次数超出阈值但last_loss仍未达标；C：迭代last_loss小于阈值。AB表示匹配失败
            #   3.1 给初始transfer
            transfer = ORIGINAL_START_TRANSFER

            #    3.2 基于初始匹配进行迭代
            map_xy_list = []
            for i in range(0, len(match_seq_list)):
                seq_start = time.time()
                match_seq = np.array(match_seq_list[i])  # 待匹配序列match_seq[N][x,y, mv, mh, PDRindex]
                start_transfer = transfer.copy()

                # 1.核心循环搜索代码
                transfer, map_xy = MMT.produce_transfer_candidates_and_search(start_transfer, TRANSFERS_PRODUCE_CONFIG,
                                                                              match_seq, mag_map,
                                                                              BLOCK_SIZE, STEP, MAX_ITERATION,
                                                                              TARGET_MEAN_LOSS,
                                                                              UPPER_LIMIT_OF_GAUSSNEWTEON,
                                                                              MMT.SearchPattern.BREAKE_ADVANCED_AND_USE_SECOND_LOSS_WHEN_FAILED)
                seq_cost_time = time.time() - seq_start
                traj_time += seq_cost_time
                all_cost_time += seq_cost_time
                max_seq_time = seq_cost_time if seq_cost_time > max_seq_time else max_seq_time

                # 4.计算该段raw_xy（仅初始对齐的PDR轨迹）\map_xy和真值iLocator_xy的误差距离，并打印输出
                index_list = []
                for p in match_seq:
                    index_list.append(p[4])
                index_list = np.array(index_list)
                index_list = index_list[:, np.newaxis]
                # 轨迹与pdr原始下标合并
                map_xy_with_index = np.concatenate((map_xy, index_list), axis=1)
                # 计算轨迹距离[dis, x,y, x,y]
                distance_of_MagPDR_iLocator_points = TEST.cal_distance_between_GT_and_MagPDR(
                    gt_xy, map_xy_with_index, xy_align_size=PDR_IMU_ALIGN_SIZE)

                seq_mean_error = np.mean(distance_of_MagPDR_iLocator_points[:, 0])
                max_win_mean_error = seq_mean_error if seq_mean_error > max_win_mean_error else max_win_mean_error

                # 修改每个滑动窗口的实际生效坐标数量
                map_xy = map_xy[0: slide_number_list[i]]
                map_xy_list.append(map_xy)

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

            pre_len = len(all_magxy_gtxy)
            for mp in magPDR_xy:
                pi = int(mp[2])
                all_magxy_gtxy.append([mp[0], mp[1],
                                       gt_xy[pi * PDR_IMU_ALIGN_SIZE][0], gt_xy[pi * PDR_IMU_ALIGN_SIZE][1]])
            # 对本条轨迹计算2个参数：MeanError TimePerM
            traj_dis = 0
            traj_magxy_gtxy = all_magxy_gtxy[pre_len:len(all_magxy_gtxy)]
            for i in range(1, len(traj_magxy_gtxy)):
                traj_dis += math.sqrt((traj_magxy_gtxy[i][2] - traj_magxy_gtxy[i - 1][2]) ** 2 + (
                        traj_magxy_gtxy[i][3] - traj_magxy_gtxy[i - 1][3]) ** 2)
            traj_error = 0
            traj_magxy_gtxy = np.array(traj_magxy_gtxy)
            for mxy, gxy in zip(traj_magxy_gtxy[:, 0:2], traj_magxy_gtxy[:, 2:4]):
                traj_error += math.sqrt((mxy[0] - gxy[0]) ** 2 + (mxy[1] - mxy[1]) ** 2)
            traj_kpi_dict[PATH_PDR_RAW[0]].append([traj_error / len(traj_magxy_gtxy), traj_time / traj_dis])
            print('\t Traj:', traj_error / len(traj_magxy_gtxy), "米", traj_time / traj_dis, "秒/米")
            # PT.paint_xy_list([traj_magxy_gtxy[:, 0:2], traj_magxy_gtxy[:, 2:4]], ['Magxy', 'GTxy'],[0, MAP_SIZE_X * 1.0, 0, MAP_SIZE_Y * 1.0])

        # 计算all_magxy_gtxy长度，计算平均耗时
        all_dis = 0
        for i in range(1, len(all_magxy_gtxy)):
            all_dis += math.sqrt((all_magxy_gtxy[i][2] - all_magxy_gtxy[i - 1][2]) ** 2 + (
                    all_magxy_gtxy[i][3] - all_magxy_gtxy[i - 1][3]) ** 2)

        cost_time_per_M = all_cost_time / all_dis
        all_error = 0
        all_magxy_gtxy = np.array(all_magxy_gtxy)
        for mxy, gxy in zip(all_magxy_gtxy[:, 0:2], all_magxy_gtxy[:, 2:4]):
            all_error += math.sqrt((mxy[0] - gxy[0]) ** 2 + (mxy[1] - mxy[1]) ** 2)
        print('\t All Trajs:', [all_error / len(all_magxy_gtxy), max_win_mean_error, cost_time_per_M, max_seq_time])
        print('\t MeanError MaxWinError CostTimePerM MaxSeqTime\n')

    for PATH_PDR_RAW in PATH_PDR_RAW_s:
        print(PATH_PDR_RAW[0])
        for d in traj_kpi_dict[PATH_PDR_RAW[0]]:
            print(d)
    return


if __name__ == '__main__':
    main()
