# 读取众包轨迹数据，统计 平均误差、误差CDF

import math
import numpy as np

files = [
    '../Paper3(MagMapBuild2)/results/XingHu/mag_q_gt_pdr.csv',
    '../Paper3(MagMapBuild2)/results/XingHu/new_mag_q_gt_pdr.csv',

    '../Paper3(MagMapBuild2)/results/InfCenter/mag_q_gt_pdr.csv',
    '../Paper3(MagMapBuild2)/results/InfCenter/new_mag_q_gt_pdr.csv'
]

def two_points_dis(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def mean_dis_bewteen_two_trajs(pdr_xy, gt_xy):
    dis_err = 0
    for pxy, gxy in zip(pdr_xy, gt_xy):
        dis_err += math.sqrt((pxy[0] - gxy[0]) ** 2 + (pxy[1] - gxy[1]) ** 2)
    return dis_err/len(gt_xy)

if __name__ == '__main__':
    # 读取gt和pdr，计算误差dis，
    old_dis_list = []
    new_dis_list = []
    for f in files:
        save_f = f + '_CDF.csv'

        data = np.loadtxt(f, delimiter=',')  # mag 0 1 2, quat 3 4 5 6, gt 7 8, pdr 9 10
        gt = data[:, 7:9]
        pdr = data[:, 9:11]

        print(f, " Mean Dis = ", mean_dis_bewteen_two_trajs(gt, pdr))

        # 计算CDF并保存
        dis_list = []
        for p1, p2 in zip(gt, pdr):
            dis = two_points_dis(p1, p2)
            dis_list.append(dis)
            if f.__contains__('new'):
                new_dis_list.append(dis)
            else:
                old_dis_list.append(dis)

        dis_list.sort()
        np.savetxt(save_f, dis_list, delimiter=',')

    print("Mean old = ", sum(old_dis_list)/len(old_dis_list))
    print("Mean new = ", sum(new_dis_list)/len(new_dis_list))
    old_dis_list.sort()
    np.savetxt('../Paper3(MagMapBuild2)/results/old_CDF.csv', old_dis_list[::10], delimiter=',')
    new_dis_list.sort()
    np.savetxt('../Paper3(MagMapBuild2)/results/new_CDF.csv', new_dis_list[::10], delimiter=',')