import numpy as np
import math
import mag_mapping_tools as MMT
import paint_tools as PT
import os

# 读取文件，第一列固定为GT（GT、NNPDR、BFS\GNI、BFS\LM、BFS\LM\EKF、MEA\EKF）
# 计算所有参数：Mean、Max、Min、50% CDF、90% CDF、RTE、TD、Traj Length
#
def cal_traj_length(traj):
    dis = 0
    for i in range(1, len(traj)):
        dis += math.sqrt((traj[i][0]-traj[i-1][0])**2+(traj[i][1]-traj[i-1][1])**2)
    return dis

if __name__ == '__main__':
    MAP_SIZE_X = 70.0
    MAP_SIZE_Y = 28.0
    all_traj_files = [
        r"../paper5/results/5/GT_NNPDR_GNI_LM_EKF1_EKF2.csv",
        r"../paper5/results/7/GT_NNPDR_GNI_LM_EKF1_EKF2.csv"
    ]
    name_list = ['GT', 'NNPDR', 'BFS\GNI', 'BFS\LM', 'BFS\LM\EKF', 'MEA\EKF']

    for traj_file in all_traj_files:
        print(traj_file)
        result_dir_path = os.path.dirname(traj_file)
        if not os.path.exists(result_dir_path):
            os.mkdir(result_dir_path)
        result_msg_file = open(result_dir_path + '/inf.txt', "w", encoding='GBK')
        all_trajs = np.loadtxt(traj_file, delimiter=',', encoding='gb2312')

        # 按照列数获取轨迹
        traj_list = []
        for j in range(0, all_trajs.shape[1], 2):
            traj_list.append(all_trajs[:, j:j + 2])

        PT.paint_xy_list(traj_list, name_list, [0, MAP_SIZE_X * 1.0, 0, MAP_SIZE_Y * 1.0])
        #
        gt = traj_list[0]
        print("Traj Len", cal_traj_length(gt))
        all_error_list = []
        for i in range(1, len(name_list)):
            error_list = []
            for gtxy, predxy in zip(gt, traj_list[i]):
                error_list.append(math.sqrt((gtxy[0] - predxy[0]) ** 2 + (gtxy[1] - predxy[1]) ** 2))
            # 统计CDF
            error_list.sort()
            all_error_list.append(error_list.copy())
            # 50% 90%
            unit = 1./len(error_list)
            cdf = 0
            cdf50 = -1
            cdf90 = -1
            for e in error_list:
                cdf += unit
                if cdf >= 0.5 and cdf50 < 0:
                    cdf50 = e
                if cdf >= 0.9 and cdf90 < 0:
                    cdf90 = e

            print(name_list[i], '\nMean={0:.3f}m, Max={1:.3f}m, Min={2:.3f}m, CDF 50%={3:.3f}m, CDF 90%={4:.3f}m'.format(
                sum(error_list) / len(error_list), max(error_list), min(error_list), cdf50, cdf90
            ))
            print(name_list[i], '\nMean={0:.3f}m, Max={1:.3f}m, Min={2:.3f}m, CDF 50%={3:.3f}m, CDF 90%={4:.3f}m'.format(
                sum(error_list) / len(error_list), max(error_list), min(error_list), cdf50, cdf90
            ), file=result_msg_file)
            np.savetxt(result_dir_path+'/CDFs.csv', np.array(all_error_list).transpose(), delimiter=',')

    # 把所有文件的data拼接到一起
    print("\n\nAll Trajs")
    all_trajs = np.loadtxt(all_traj_files[0], delimiter=',')
    for i in range(1, len(all_traj_files)):
        all_trajs = np.concatenate((all_trajs, np.loadtxt(all_traj_files[1], delimiter=',')), axis=0)

    # 按照列数获取轨迹
    traj_list = []
    for j in range(0, all_trajs.shape[1], 2):
        traj_list.append(all_trajs[:, j:j + 2])
    gt = traj_list[0]
    for i in range(1, len(name_list)):
        error_list = []
        for gtxy, predxy in zip(gt, traj_list[i]):
            error_list.append(math.sqrt((gtxy[0] - predxy[0]) ** 2 + (gtxy[1] - predxy[1]) ** 2))
        # 统计CDF
        error_list.sort()
        # 50% 90%
        unit = 1. / len(error_list)
        cdf = 0
        cdf50 = -1
        cdf90 = -1
        for e in error_list:
            cdf += unit
            if cdf >= 0.5 and cdf50 < 0:
                cdf50 = e
            if cdf >= 0.9 and cdf90 < 0:
                cdf90 = e

        print(name_list[i],
              '\nMean={0:.3f}m, Max={1:.3f}m, Min={2:.3f}m, CDF 50%={3:.3f}m, CDF 90%={4:.3f}m'.format(
                  sum(error_list) / len(error_list), max(error_list), min(error_list), cdf50, cdf90
              ))
