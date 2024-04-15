# 读取全部用于建库的数据
# 将它们进行合并，分别绘图NNPDR和GT的图
import mag_mapping_tools as MMT
import numpy as np
import paint_tools as PT

# 信息中心
file_paths = [
    # 1
    "../data/InfCenter server room/mag_map_build/1/IMU-812-1-277.9703774644415 Pixel 6_sync.csv",
    "../data/InfCenter server room/mag_map_build/2/IMU-812-2-269.1133706393625 Pixel 6_sync.csv",
    "../data/InfCenter server room/mag_map_build/3/IMU-812-3-280.63754584476396 Pixel 6_sync.csv",
    "../data/InfCenter server room/mag_map_build/4/IMU-812-4-274.7988610856161 Pixel 6_sync.csv",

    # 2
    "../data/InfCenter server room/position_test/5/IMU-812-5-277.2496012617084 Pixel 6_sync.csv",
    "../data/InfCenter server room/position_test/6/IMU-812-6-269.09426660025395 Pixel 6_sync.csv",
    "../data/InfCenter server room/position_test/7/IMU-812-7-195.4948665194862 Pixel 6_sync.csv",
    "../data/InfCenter server room/position_test/8/IMU-812-8-193.38120983931242 Pixel 6_sync.csv",
    "../data/InfCenter server room/position_test/9/IMU-812-9-189.79622112889115 Pixel 6_sync.csv"
]

# file_paths = [
#     # 1
#     "../data/XingHu hall 8F test/mag_map_build/1/IMU-88-1-155.58859472662894 Pixel 6_sync.csv",
#     "../data/XingHu hall 8F test/mag_map_build/2/IMU-88-2-181.05721265191326 Pixel 6_sync.csv",
#     "../data/XingHu hall 8F test/mag_map_build/3/IMU-88-3-276.6560168416993 Pixel 6_sync.csv",
#     "../data/XingHu hall 8F test/mag_map_build/4/IMU-88-4-201.15885553023264 Pixel 6_sync.csv",
#
#     # 2
#     "../data/XingHu hall 8F test/position_test/5/IMU-88-5-291.0963959547511 Pixel 6_sync.csv",
#     "../data/XingHu hall 8F test/position_test/6/IMU-88-6-194.9837361431375 Pixel 6_sync.csv",
#     "../data/XingHu hall 8F test/position_test/7/IMU-88-7-270.6518297687728 Pixel 6_sync.csv",
#     "../data/XingHu hall 8F test/position_test/8/IMU-88-8-189.88230883318997 Pixel 6_sync.csv"
# ]

pdr_xy_files = [
    "../data/InfCenter server room/mag_map_build/1/IMU-812-1-277.9703774644415 Pixel 6_sync.csv.npy",
    "../data/InfCenter server room/mag_map_build/2/IMU-812-2-269.1133706393625 Pixel 6_sync.csv.npy",
    "../data/InfCenter server room/mag_map_build/3/IMU-812-3-280.63754584476396 Pixel 6_sync.csv.npy",
    "../data/InfCenter server room/mag_map_build/4/IMU-812-4-274.7988610856161 Pixel 6_sync.csv.npy",

    # 2
    "../data/InfCenter server room/position_test/5/IMU-812-5-277.2496012617084 Pixel 6_sync.csv.npy",
    "../data/InfCenter server room/position_test/6/IMU-812-6-269.09426660025395 Pixel 6_sync.csv.npy",
    "../data/InfCenter server room/position_test/7/IMU-812-7-195.4948665194862 Pixel 6_sync.csv.npy",
    "../data/InfCenter server room/position_test/8/IMU-812-8-193.38120983931242 Pixel 6_sync.csv.npy",
    "../data/InfCenter server room/position_test/9/IMU-812-9-189.79622112889115 Pixel 6_sync.csv.npy"
]

# pdr_xy_files = [
#     # 1
#     "../data/XingHu hall 8F test/mag_map_build/1/IMU-88-1-155.58859472662894 Pixel 6_sync.csv.npy",
#     "../data/XingHu hall 8F test/mag_map_build/2/IMU-88-2-181.05721265191326 Pixel 6_sync.csv.npy",
#     "../data/XingHu hall 8F test/mag_map_build/3/IMU-88-3-276.6560168416993 Pixel 6_sync.csv.npy",
#     "../data/XingHu hall 8F test/mag_map_build/4/IMU-88-4-201.15885553023264 Pixel 6_sync.csv.npy",
#
#     # 2
#     "../data/XingHu hall 8F test/position_test/5/IMU-88-5-291.0963959547511 Pixel 6_sync.csv.npy",
#     "../data/XingHu hall 8F test/position_test/6/IMU-88-6-194.9837361431375 Pixel 6_sync.csv.npy",
#     "../data/XingHu hall 8F test/position_test/7/IMU-88-7-270.6518297687728 Pixel 6_sync.csv.npy",
#     "../data/XingHu hall 8F test/position_test/8/IMU-88-8-189.88230883318997 Pixel 6_sync.csv.npy"
# ]

# 信息中心
MOVE_X = 5.
MOVE_Y = 5.
MAP_SIZE_X = 35.
MAP_SIZE_Y = 20.

# 星湖楼
# MOVE_X = 10.0
# MOVE_Y = 15.0
# MAP_SIZE_X = 70.0
# MAP_SIZE_Y = 28.0

if __name__ == '__main__':
    mag_q_gt_pdr = []
    for f_all, f_pdr in zip(file_paths, pdr_xy_files):

        data_all = MMT.get_data_from_csv(f_all)

        mag = data_all[:, 21:24]
        quat = data_all[:, 7:11]  # GAME_ROTATION_VECTOR 未经磁场矫正的旋转向量（四元数）
        gt_xy = data_all[:, np.shape(data_all)[1] - 5:np.shape(data_all)[1] - 3]  # gt_xy 用于判断行为地标，数量等于IMU数量
        print(gt_xy[0])

        pdr_xy = np.load(f_pdr)[:, 0:2]  # TODO pdr_xy 用于构建众包指纹地图，需要和gt_xy根据10进行对齐，先对齐后再合并！因为每个文件后面可能会少一个窗口的IMU数据

        for i in range(0, len(pdr_xy)):
            j = 10 * i
            mag_q_gt_pdr.append([
                mag[j][0], mag[j][1], mag[j][2],
                quat[j][0], quat[j][1], quat[j][2], quat[j][3],
                gt_xy[j][0] + MOVE_X, gt_xy[j][1] + MOVE_Y,
                pdr_xy[i][0] + MOVE_X, pdr_xy[i][1] + MOVE_Y
            ])

    # 合并文件 mag 0 1 2, quat 3 4 5 6, gt 7 8, pdr 9 10
    mag_q_gt_pdr = np.array(mag_q_gt_pdr)
    # np.savetxt("../Paper3(MagMapBuild2)/results/XingHu/mag_q_gt_pdr.csv", mag_q_gt_pdr, delimiter=',')
    np.savetxt("../Paper3(MagMapBuild2)/results/InfCenter/mag_q_gt_pdr.csv", mag_q_gt_pdr, delimiter=',')

    # 绘制NNPDR轨迹图，GT轨迹图
    XingHu_map_size = [0, MAP_SIZE_X * 1, 0, MAP_SIZE_Y * 1]
    PT.paint_xy_list([mag_q_gt_pdr[:, 7:9]], ['GT'], XingHu_map_size, '')
    PT.paint_xy_list([mag_q_gt_pdr[:, 9:11]], ['PDR'], XingHu_map_size, '')
    PT.paint_xy_list([mag_q_gt_pdr[:, 7:9], mag_q_gt_pdr[:, 9:11]], ['PDR', 'GT'], XingHu_map_size, '')
