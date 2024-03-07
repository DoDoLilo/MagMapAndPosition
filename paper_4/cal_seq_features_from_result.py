import mag_features as MF
import mag_mapping_tools as MMT
import math
import numpy as np

def max_min_to_01(seq):
    max_ = max(seq)
    min_ = min(seq)
    new_seq = []
    for d in seq:
        new_seq.append((d-min_)/(max_-min_))
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

    disXy_mvhSeq_initByGT = []  # [N][0:dis, 1:mX, 2:mY, 3:gtX, 4:gtY, 5:mv, 6:mh]
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
    win_size = int(5 / block_size)
    slide_size = 2
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

        # RMSD
        RMSD = 0
        for dis in disXy_mvhSeq_initByGT[si:si + win_size, 0]:
            RMSD += dis ** 2
        RMSD = math.sqrt(RMSD / win_size)

        # 注意xgx和其他特征与RMSD的关系是反的
        save_data.append([mean_fc, mean_td, mean_xgx, RMSD])

        si += slide_size

    save_data = np.array(save_data)
    # TODO 将特征进行Max-Min归一化
    new_save_data = []
    for d in zip(max_min_to_01(save_data[:, 0]), max_min_to_01(save_data[:, 1]),max_min_to_01(save_data[:, 2]),save_data[:, 3]):
        new_save_data.append(d)

    # TODO 把底部密集的三角形区域的数据按照随机概率删除一部分

    # 保存结果为csv
    # np.savetxt("../paper_4/results/XinghuHall_seq_features_RMSD_all_inReal.csv", new_save_data, delimiter=',')

    # np.savetxt("../paper_4/results/InfCenter_seq_features_RMSD_all_initByGT.csv", new_save_data, delimiter=',')
    # np.savetxt("../paper_4/results/XinghuHall_seq_features_RMSD_all_initByGT.csv", new_save_data, delimiter=',')
    np.savetxt("../paper_4/results/All_seq_features_RMSD_all_initByGT.csv", new_save_data, delimiter=',')
