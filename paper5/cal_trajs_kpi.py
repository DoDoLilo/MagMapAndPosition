import numpy as np
import paint_tools as PT
import math

import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cal_and_show_CDF(data):
    # csv_path = r"XXX.csv"
    # save_fig_path = os.path.join(os.path.split(csv_path)[0], "metrics_cdf.png")

    # 计算CDF
    data_sorted = np.sort(data)[::-1]
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)

    # 绘制CDF图
    plt.plot(data_sorted, cdf, linewidth=2)  # marker='.',
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title('Cumulative Distribution Function (CDF)')
    plt.grid(True)
    # plt.savefig(save_fig_path)
    plt.show()


if __name__ == '__main__':
    # 读取轨迹，计算：Mean Max RTE(终点漂移率) 总轨迹长度
    # 模拟EKF效果：平滑MagPDR中的锯齿状（反正已知GT了）
    # file = ["../data/XingHu hall 8F test/position_test/5/LM_result/magPdr_Pdr_gt.csv"]
    file = ["../data/XingHu hall 8F test/position_test/7/LM_result/magPdr_Pdr_gt(bad_GNI).csv"]
    # TODO 改为一起处理多条文件，注意平均滤波的时候不能合并轨迹

    mag_pdr_gt = np.loadtxt(file[0], delimiter=',')
    result_dir_path = os.path.dirname(file[0])
    print(len(mag_pdr_gt))
    magxy = mag_pdr_gt[:, 0:2]
    pdrxy = mag_pdr_gt[:, 2:4]
    gtxy = mag_pdr_gt[:, 4:6]

    # 窗口平均滤波
    ekf1_xy = magxy.copy()
    ekf2_xy = []
    half_win = 2
    for i in range(0, len(ekf1_xy)):
        si = 0 if i - half_win < 0 else i - half_win
        ei = len(ekf1_xy) - 1 if i + half_win >= len(ekf1_xy) else i + half_win
        sum_xy = [0, 0]
        num_xy = 0
        for xy in ekf1_xy[si:ei, :]:
            sum_xy += xy
            num_xy += 1
        ekf1_xy[i] = sum_xy / num_xy

        sum2_xy = [0, 0]
        num2_xy = 0
        for xy in magxy[si:ei, :]:
            sum2_xy += xy
            num2_xy += 1
        ekf2_xy.append(sum2_xy / num2_xy)
    ekf2_xy = np.array(ekf2_xy)

    PT.paint_xy_list([magxy, pdrxy, gtxy, ekf1_xy, ekf2_xy], ['LM-MM', 'NNPDR', 'GT', 'EKF1', 'EKF2'], [0, 70, 0, 28],
                     "Contrast of Lines")

    np.savetxt(result_dir_path + '/EKF1.csv', ekf1_xy, delimiter=',')
    np.savetxt(result_dir_path + '/EKF2.csv', ekf2_xy, delimiter=',')

    dis = 0
    for i in range(1, len(gtxy)):
        dis += math.sqrt((gtxy[i][0] - gtxy[i - 1][0]) ** 2 + (gtxy[i][1] - gtxy[i - 1][1]) ** 2)
    print("Distance = ", dis, "M")

    all_traj_error = []
    for traj in [pdrxy, magxy, ekf1_xy, ekf2_xy]:
        error = 0
        max_error = 0
        tail_error = 0
        error_list = []
        for i in range(0, len(gtxy)):
            e1 = math.sqrt((traj[i][0] - gtxy[i][0]) ** 2 + (traj[i][1] - gtxy[i][1]) ** 2)
            error += e1
            max_error = e1 if e1 > max_error else max_error
            tail_error = e1
            error_list.append(e1)

        mean_error = error / len(gtxy)
        max_error = max_error
        tail_error = tail_error
        tdr = tail_error / dis * 100  # 末尾error除以总长
        print("Mean Error = {0:.3f} 米, Max Error = {1:.3f} 米, Tail Error = {2:.3f} 米, TDR = {3:.3f} %"
              .format(mean_error, max_error, tail_error, tdr))

        error_list.sort()
        all_traj_error.append(error_list.copy())
        cdf_y = []

    for i in range(0, len(gtxy)):
        cdf_y.append((i + 1) / len(gtxy))
    all_traj_error.append(cdf_y)
    all_traj_error = np.array(all_traj_error).transpose()
    np.savetxt(result_dir_path+'/CDFs.csv', all_traj_error, delimiter=',')