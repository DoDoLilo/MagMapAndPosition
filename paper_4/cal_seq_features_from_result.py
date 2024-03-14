import mag_features as MF
import mag_mapping_tools as MMT
import math
import numpy as np
import random


def max_min_to_01(seq):
    max_ = max(seq)
    min_ = min(seq)
    new_seq = []
    for d in seq:
        new_seq.append((d - min_) / (max_ - min_))
    return new_seq


if __name__ == '__main__':
    result_files = [
        "../data/XingHu hall 8F test/position_test/5/paper4_result/disXy_mvhSeq_initByGT.csv",
        "../data/XingHu hall 8F test/position_test/6/paper4_result/disXy_mvhSeq_initByGT.csv",
        "../data/XingHu hall 8F test/position_test/7/paper4_result/disXy_mvhSeq_initByGT.csv",
        "../data/XingHu hall 8F test/position_test/8/paper4_result/disXy_mvhSeq_initByGT.csv",

        "../data/InfCenter server room/position_test/5/paper4_result/disXy_mvhSeq_initByGT.csv",
        "../data/InfCenter server room/position_test/6/paper4_result/disXy_mvhSeq_initByGT.csv",
        "../data/InfCenter server room/position_test/7/paper4_result/disXy_mvhSeq_initByGT.csv",
        "../data/InfCenter server room/position_test/8/paper4_result/disXy_mvhSeq_initByGT.csv",
        "../data/InfCenter server room/position_test/9/paper4_result/disXy_mvhSeq_initByGT.csv"
    ]

    # result_files = [
    # "../data/XingHu hall 8F test/position_test/5/paper4_result/disXy_mvhSeq_inReal.csv",
    # "../data/XingHu hall 8F test/position_test/6/paper4_result/disXy_mvhSeq_inReal.csv",
    # "../data/XingHu hall 8F test/position_test/7/paper4_result/disXy_mvhSeq_inReal.csv",
    # "../data/XingHu hall 8F test/position_test/8/paper4_result/disXy_mvhSeq_inReal.csv"

    # "../data/InfCenter server room/position_test/5/paper4_result/disXy_mvhSeq_inReal.csv",
    # "../data/InfCenter server room/position_test/6/paper4_result/disXy_mvhSeq_inReal.csv",
    # "../data/InfCenter server room/position_test/7/paper4_result/disXy_mvhSeq_inReal.csv",
    # "../data/InfCenter server room/position_test/8/paper4_result/disXy_mvhSeq_inReal.csv",
    # "../data/InfCenter server room/position_test/9/paper4_result/disXy_mvhSeq_inReal.csv"
    # ]

    disXy_mvhSeq_initByGT = []  # [N][0:dis, 1:mX, 2:mY, 3:gtX, 4:gtY, 5:mv, 6:mh, 7:area_fc, 8:area_td,9:area_xgx]
    for f in result_files:
        result = MMT.get_data_from_csv(f)
        for d1 in result:
            temp = []
            for d2 in d1:
                temp.append(d2)
            disXy_mvhSeq_initByGT.append(temp)

    disXy_mvhSeq_initByGT = np.array(disXy_mvhSeq_initByGT)

    # 使用滑窗的形式，切分 dis与特征seq，计算RMSD与特征均值
    block_size = 0.3
    win_size = int(6 / block_size)
    slide_size = 3
    si = 0

    # 先计算全部的特征序列
    mv_seq = disXy_mvhSeq_initByGT[:, 5]
    mh_seq = disXy_mvhSeq_initByGT[:, 6]
    mall_seq = []
    for mv, mh in zip(mv_seq, mh_seq):
        mall_seq.append(math.sqrt(mv ** 2 + mh ** 2))

    fc_seq, td_seq, xgx_seq = MF.get_seq_feature_seqs(mall_seq, 4)
    print(len(disXy_mvhSeq_initByGT), ",", len(fc_seq), ",", len(td_seq), ",", len(xgx_seq), ",", len(mall_seq))

    save_data = []
    while si + win_size < len(fc_seq):
        # 特征均值
        mean_fc = sum(fc_seq[si:si + win_size]) / win_size
        mean_td = sum(td_seq[si:si + win_size]) / win_size
        mean_xgx = sum(xgx_seq[si:si + win_size]) / win_size

        mean_area_fc = sum(disXy_mvhSeq_initByGT[si:si + win_size, 7]) / win_size
        mean_area_td = sum(disXy_mvhSeq_initByGT[si:si + win_size, 8]) / win_size
        mean_area_xgx = sum(disXy_mvhSeq_initByGT[si:si + win_size, 9]) / win_size


        # RMSD
        RMSD = 0
        for dis in disXy_mvhSeq_initByGT[si:si + win_size, 0]:
            RMSD += dis ** 2
        RMSD = math.sqrt(RMSD / win_size)

        # 注意xgx和其他特征与RMSD的关系是反的
        save_data.append([mean_fc, mean_td, -mean_xgx, mean_area_fc, mean_area_td, -mean_area_xgx, RMSD])

        si += slide_size

    save_data = np.array(save_data)
    # 将特征进行Max-Min归一化
    new_save_data = []
    for d in zip(max_min_to_01(save_data[:, 0]), max_min_to_01(save_data[:, 1]), max_min_to_01(save_data[:, 2]),
                 max_min_to_01(save_data[:, 3]), max_min_to_01(save_data[:, 4]), max_min_to_01(save_data[:, 5]),
                 save_data[:, 6]):
        new_save_data.append(d)


    a = 7
    b = 0

    # 遍历new_save_data中的3个特征与RMSD，RMSD为 y
    deleted_result = []
    deleted_p1 = 85
    deleted_p2 = 90
    deleted_p3 = 95
    for i in range(0, 3):
        temp1 = []
        for d in new_save_data:
            f = d[i]  # 特征是被归一化的，所以是x
            r = d[6]  # 是 y
            # 判断是否落在函数下方 and 随机删除p%的（3级概率）
            if r < (1/(1+(a*8)*f)-b) and random.randint(1, 101) <= deleted_p3:
                continue
            elif r < (1/(1+(a*2.5)*f)-b) and random.randint(1, 101) <= deleted_p2:
                continue
            elif r < (1/(1+a*f)-b) and random.randint(1, 101) <= deleted_p1:
                continue
            else:
                # 添加该特征与rmsd到数组
                temp1.append([f, r])
        print(len(temp1))
        deleted_result.append(temp1)

    # 因为可能长度不一样了，需要分别进行保存
    # np.savetxt("../paper_4/results/All_seq_features_RMSD_all_initByGT(deleted)/fc.csv", deleted_result[0],
    #            delimiter=',')
    # np.savetxt("../paper_4/results/All_seq_features_RMSD_all_initByGT(deleted)/td.csv", deleted_result[1],
    #            delimiter=',')
    # np.savetxt("../paper_4/results/All_seq_features_RMSD_all_initByGT(deleted)/xgx.csv", deleted_result[2],
    #            delimiter=',')

    for i in range(3, 6):
        temp1 = []
        for d in new_save_data:
            f = d[i]  # 特征是被归一化的，所以是x
            r = d[6]  # 是 y
            # 判断是否落在函数下方 and 随机删除p%的（3级概率）
            if (r < (1/(0.5+14*f)-0.2) ) and random.randint(1, 101) <= deleted_p3:
                continue
            elif (r < (1/(0.5+5*f)-0.2) ) and random.randint(1, 101) <= deleted_p2:
                continue
            elif (r < (1/(0.5+3*f)-0.2) or r>(1/(0.5+3*(f-0.2)))) and random.randint(1, 101) <= deleted_p1:
                continue
            else:
                # 添加该特征与rmsd到数组
                temp1.append([f, r])
        print(len(temp1))
        deleted_result.append(temp1)


    np.savetxt("../paper_4/results/All_area_features_RMSD_all_initByGT(deleted)/fc.csv", deleted_result[3],
               delimiter=',')
    np.savetxt("../paper_4/results/All_area_features_RMSD_all_initByGT(deleted)/td.csv", deleted_result[4],
               delimiter=',')
    np.savetxt("../paper_4/results/All_area_features_RMSD_all_initByGT(deleted)/xgx.csv", deleted_result[5],
               delimiter=',')

    # 保存结果为csv
    # np.savetxt("../paper_4/results/XinghuHall_seq_features_RMSD_all_inReal.csv", new_save_data, delimiter=',')

    # np.savetxt("../paper_4/results/InfCenter_seq_features_RMSD_all_initByGT.csv", new_save_data, delimiter=',')
    # np.savetxt("../paper_4/results/XinghuHall_seq_features_RMSD_all_initByGT.csv", new_save_data, delimiter=',')
    np.savetxt("../paper_4/results/All_features_RMSD_all_initByGT.csv", new_save_data, delimiter=',')
