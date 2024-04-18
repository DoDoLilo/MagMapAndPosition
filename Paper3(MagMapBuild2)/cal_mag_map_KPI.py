# 读取俩磁场指纹地图，计算对应网格磁强差异
import numpy as np
import paint_tools as PT
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heat_map(mag_map):
    # plt.figure(figsize=(10, 36))
    plt.figure(figsize=(20, 10))
    # sns.set(font_scale=0.8)
    # cmap='YlOrRd'
    sns.heatmap(
        data=mag_map[:, ::-1].T,
        cmap='jet',
        # annot=True, fmt='.0f',
        mask=mag_map[:, ::-1].T == -1,
        cbar=True, vmax=70, vmin=-1
    )
    plt.show()

if __name__ == '__main__':


    # InfCenter_maps = [
    #     '../Paper3(MagMapBuild2)/results/InfCenter/map_F1_2_3_4_B_0.3_deleted/ms.csv',
    #     '../Paper3(MagMapBuild2)/results/InfCenter/map_F_deleted/ms.csv'
    # ]

    # InfCenter_maps = [
    #     '../Paper3(MagMapBuild2)/results/InfCenter/map_F1_2_3_4_B_0.3_deleted/ms.csv',
    #     '../Paper3(MagMapBuild2)/results/InfCenter/map_F_deleted/ms.csv'
    # ]
    #
    # save_dir = '../Paper3(MagMapBuild2)/results/InfCenter/map_F_deleted/'
    #
    # map_gt = np.loadtxt(InfCenter_maps[0], delimiter=',')
    # map_pdr = np.loadtxt(InfCenter_maps[1], delimiter=',')

    # XingHu_maps = [
    #     '../Paper3(MagMapBuild2)/results/XingHu/map_F1_2_3_4_5_6_7_8_B_0.3_deleted/ms.csv',
    #     '../Paper3(MagMapBuild2)/results/XingHu/map_F_deleted/ms.csv'
    # ]

    XingHu_maps = [
        '../Paper3(MagMapBuild2)/results/XingHu/map_F1_2_3_4_5_6_7_8_B_0.3_deleted/ms.csv',
        '../Paper3(MagMapBuild2)/results/XingHu/map_F_deleted/ms.csv'
    ]



    save_dir = '../Paper3(MagMapBuild2)/results/XingHu/map_F_deleted/'

    map_gt = np.loadtxt(XingHu_maps[0], delimiter=',')
    map_pdr = np.loadtxt(XingHu_maps[1], delimiter=',')

    plot_heat_map(map_gt)
    plot_heat_map(map_pdr)

    mag_errors = []  # [N][mag_dis, ms_gt, ms_pdr]

    for i in range(0, len(map_gt)):
        for j in range(0, len(map_gt[0])):
            ms_gt, ms_pdr = map_gt[i][j], map_pdr[i][j]
            if ms_gt >= 0 and ms_pdr >= 0:
                # 俩网格都得有数据才行
                mag_errors.append([abs(ms_gt - ms_pdr), ms_gt, ms_pdr])

    # 根据磁场差异距离，对其进行排序
    mag_errors = np.array(mag_errors)
    sorted_indices = np.argsort(mag_errors[:, 0])
    mag_errors = mag_errors[sorted_indices]

    print("Max Dis = {0}, Min = {1}, Mean = {2}, CDF 50% = {3}, CDF 90% = {4}".format(
        max(mag_errors[:, 0]), min(mag_errors[:, 0]), sum(mag_errors[:, 0]) / len(mag_errors[:, 0]),
        mag_errors[int(len(mag_errors) * 0.5), 0], mag_errors[int(len(mag_errors) * 0.9), 0]
    ))

    max_e_i = 0
    min_e_i = 0
    for i in range(0, len(mag_errors)):
        if mag_errors[i, 0] == max(mag_errors[:, 0]):
            max_e_i = i
        if mag_errors[i, 0] == min(mag_errors[:, 0]):
            min_e_i = i

    print("Percent of Max Dis = {0}%, Min = {1}%, Mean = {2}%".format(
        max(mag_errors[:, 0])*100/mag_errors[max_e_i, 1], min(mag_errors[:, 0])*100/mag_errors[min_e_i, 1],
        (sum(mag_errors[:, 0]) / len(mag_errors[:, 0]))*100/(sum(mag_errors[:, 1]) / len(mag_errors[:, 1]))
    ))

    # np.savetxt(save_dir+'mag_errors(CMM).csv', mag_errors, delimiter=',')

    # 误差百分比 = error / ms_gt
    errors_percent = []
    for line in errors_percent:
        e = line[0]
        gt = line[1]
        errors_percent.append(e*100/gt)

    errors_percent = np.array(errors_percent)
    errors_percent.sort()
    # np.savetxt(save_dir+'error_percent(CMM).csv', errors_percent, delimiter=',')



