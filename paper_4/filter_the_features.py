import mag_features as MF
import mag_mapping_tools as MMT
import math
import numpy as np
import random

if __name__ == '__main__':
    # 读取论文4.3.2节的实验数据结果，根据RMSD进行特征合并，用于多元线性回归
    # Seq方差,RMSD + Seq梯度,RMSD + Area方差,RMSD + Area梯度,RMSD

    fs_file = "../paper_4/results/All_features_RMSD_all_initByGT(paper).csv"
    data = np.loadtxt(fs_file, delimiter=',')

    seq_fc = data[:, 0:2]
    seq_td = data[:, 2:4]
    area_fc = data[:, 4:6]
    area_td = data[:, 6:8]

    # 按照一定容忍范围，在数据中查找相同的RMSD，保存RMSD的均值和对应的4组特征值{seq_fc, seq_td, area_fc, area_td}
    same_range = 1/100
    same_num = 0

    rmsd_fs = []  # [N][5] = RMSD, seq_fc, seq_td, area_fc, area_td
    for rmsd_0, sf in seq_fc:
        st = -1
        for rmsd_1, f in seq_td:
            if abs(rmsd_1-rmsd_0) < same_range:
                st = f
                break

        af = -1
        for rmsd_2, f in area_fc:
            if abs(rmsd_2 - rmsd_0) < same_range:
                af = f
                break

        at = -1
        for rmsd_3, f in area_td:
            if abs(rmsd_3-rmsd_0) < same_range:
                at = f
                break

        if not (sf < 0 or st < 0 or af < 0 or at < 0):
            same_num += 1
            rmsd_fs.append([(rmsd_0+rmsd_1+rmsd_2+rmsd_3)/4, sf, st, af, at])

    for rmsd_0, af in area_fc:
        st = -1
        for rmsd_1, f in seq_td:
            if abs(rmsd_1-rmsd_0) < same_range:
                st = f
                break

        sf = -1
        for rmsd_2, f in seq_fc:
            if abs(rmsd_2 - rmsd_0) < same_range:
                sf = f
                break

        at = -1
        for rmsd_3, f in area_td:
            if abs(rmsd_3-rmsd_0) < same_range:
                at = f
                break

        if not (sf < 0 or st < 0 or af < 0 or at < 0):
            same_num += 1
            rmsd_fs.append([(rmsd_0+rmsd_1+rmsd_2+rmsd_3)/4, sf, st, af, at])


    # 重复4组，分别选取不同的特征作为 rmsd_0
    print("Data num = ", same_num)
    np.savetxt("../paper_4/results/RMSD_sf_st_af_at(MatLab).csv", rmsd_fs, delimiter=',')