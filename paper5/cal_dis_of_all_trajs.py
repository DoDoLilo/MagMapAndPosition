import numpy as np
import mag_mapping_tools as MMT
import math

def cal_trajs_dis(files):
    all_dis = 0
    for f in files:
        print(f[0])
        data_all = MMT.get_data_from_csv(f[1])
        gt_xy = data_all[:, np.shape(data_all)[1] - 5:np.shape(data_all)[1] - 3]
        dis = 0
        for i in range(1, len(gt_xy)):
            dis += math.sqrt((gt_xy[i][0]-gt_xy[i-1][0])**2+(gt_xy[i][1]-gt_xy[i-1][1])**2)
        print(dis)
        all_dis += dis
    print("All Mean Dis = ", all_dis/len(files))


if __name__ == '__main__':
    trajs_files = [
        ['../data/XingHu hall 8F test/position_test/5/IMU-88-5-291.0963959547511 Pixel 6_sync.csv.npy',
         '../data/XingHu hall 8F test/position_test/5/IMU-88-5-291.0963959547511 Pixel 6_sync.csv'],
        ['../data/XingHu hall 8F test/position_test/6/IMU-88-6-194.9837361431375 Pixel 6_sync.csv.npy',
         '../data/XingHu hall 8F test/position_test/6/IMU-88-6-194.9837361431375 Pixel 6_sync.csv'],
        ['../data/XingHu hall 8F test/position_test/7/IMU-88-7-270.6518297687728 Pixel 6_sync.csv.npy',
         '../data/XingHu hall 8F test/position_test/7/IMU-88-7-270.6518297687728 Pixel 6_sync.csv'],
        ['../data/XingHu hall 8F test/position_test/8/IMU-88-8-189.88230883318997 Pixel 6_sync.csv.npy',
         '../data/XingHu hall 8F test/position_test/8/IMU-88-8-189.88230883318997 Pixel 6_sync.csv'],

        [
            "../data/InfCenter server room/position_test/5/IMU-812-5-277.2496012617084 Pixel 6_sync.csv.npy",
            "../data/InfCenter server room/position_test/5/IMU-812-5-277.2496012617084 Pixel 6_sync.csv"],
        [
            "../data/InfCenter server room/position_test/6/IMU-812-6-269.09426660025395 Pixel 6_sync.csv.npy",
            "../data/InfCenter server room/position_test/6/IMU-812-6-269.09426660025395 Pixel 6_sync.csv"],
        [
            "../data/InfCenter server room/position_test/7/IMU-812-7-195.4948665194862 Pixel 6_sync.csv.npy",
            "../data/InfCenter server room/position_test/7/IMU-812-7-195.4948665194862 Pixel 6_sync.csv"],
        [
            "../data/InfCenter server room/position_test/8/IMU-812-8-193.38120983931242 Pixel 6_sync.csv.npy",
            "../data/InfCenter server room/position_test/8/IMU-812-8-193.38120983931242 Pixel 6_sync.csv"],
        [
            "../data/InfCenter server room/position_test/9/IMU-812-9-189.79622112889115 Pixel 6_sync.csv.npy",
            "../data/InfCenter server room/position_test/9/IMU-812-9-189.79622112889115 Pixel 6_sync.csv"]
    ]

    # 星湖楼4条
    print("XingHu Hall")
    cal_trajs_dis(trajs_files[0:4])
    print("\nInfCenter")
    cal_trajs_dis(trajs_files[4:])
