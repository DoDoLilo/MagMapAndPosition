import math

import mag_mapping_tools as MMT
import numpy as np

MOVE_X = 7.
MOVE_Y = 8.
MAP_SIZE_X = 8
MAP_SIZE_Y = 13
BLOCK_SIZE = 0.25
INTER_RADIUS = 1
INTER_TIME_THR = 2
DELETE_LEVEL = 0
# ------------------------
BUFFER_DIS = 2
DOWN_SIP_DIS = BLOCK_SIZE
MAX_ITERATION = 8
TARGET_LOSS = 0
STEP = 1/6000

def main():
    # 全流程
    # 1、建库（未滤波）
    file_paths_build_map = [
        "data/data_test/data_to_building_map/IMU-10-1-190.80648806940607 Pixel 6_sync.csv",
        "data/data_test/data_to_building_map/IMU-10-2-183.5307793202117 Pixel 6_sync.csv",
        "data/data_test/data_to_building_map/IMU-10-3-170.97105500171142 Pixel 6_sync.csv",
        "data/data_test/data_to_building_map/IMU-10-4-180.40767532222338 Pixel 6_sync.csv",
        "data/data_test/data_to_building_map/IMU-10-5-170.2125898151382 Pixel 6_sync.csv",
        "data/data_test/data_to_building_map/IMU-10-6-178.00767980919863 Pixel 6_sync.csv"
    ]

    mag_map = MMT.build_map_by_files(
        file_paths=file_paths_build_map,
        move_x=MOVE_X, move_y=MOVE_Y,
        map_size_x=MAP_SIZE_X, map_size_y=MAP_SIZE_Y,
        # time_thr=INTER_TIME_THR,
        radius=INTER_RADIUS, block_size=BLOCK_SIZE,
        delete=False,
        delete_level=DELETE_LEVEL
    )

    # 2、缓冲池给匹配段（内置稀疏采样），此阶段的data与上阶段无关
    file_to_match = "data/data_test/data_to_building_map/IMU-10-1-190.80648806940607 Pixel 6_sync.csv"
    pdr_data_all = MMT.get_data_from_csv(file_to_match)
    pdr_data_mag = pdr_data_all[:, 21:24]
    pdr_data_ori = pdr_data_all[:, 18:21]
    pdr_data_xy = pdr_data_all[:, np.shape(pdr_data_all)[1] - 5:np.shape(pdr_data_all)[1] - 3]
    # 并不在此时修改pdr_xy坐标，match_seq_list=多条匹配序列[?][?][x,y, mv, mh]
    match_seq_list = MMT.samples_buffer(BUFFER_DIS, DOWN_SIP_DIS, pdr_data_ori, pdr_data_mag, pdr_data_xy)

    # 3、手动给出初始transfer_0，注意单条
    # 根据匹配段进行迭代，3种迭代结束情况：
    #  A：迭代out_of_map返回True；B：迭代次数超出阈值但last_loss仍未达标；C：迭代last_loss小于阈值
    # 情况AB表示匹配失败，论文未说匹配失败的解决方法
    transfer = np.array([0., 7., math.radians(90.)])
    seq_num = 0
    # TODO 第一次匹配进行全图遍历
    for match_seq in match_seq_list:
        match_seq = np.array(match_seq)
        seq_num += 1
        print("\nMatch Seq:{0}   --------------------------".format(seq_num))
        # 单次轨迹地磁匹配 do while
        print(match_seq[:, 0:2])
        iter_num = 1
        out_of_map, loss, map_xy, transfer = MMT.cal_new_transfer_and_last_loss_xy(
            transfer, match_seq, mag_map, BLOCK_SIZE, STEP
        )
        while True:
            print("  iteration {0}: OutOfMap={1}, Loss={2}, Transfer={3}".format(
                iter_num, out_of_map, loss, transfer))
            print(map_xy)
            if out_of_map or iter_num > MAX_ITERATION:
                print("Match Seq:", seq_num, "iteration ", iter_num, ", Failed!")
                # TODO 失败了怎么办？
                break
            if loss <= TARGET_LOSS:
                print("Match Seq:", seq_num, "iteration ", iter_num, ", Succeed!")
                # TODO 成功了怎么办？
                break
            iter_num += 1
            out_of_map, loss, map_xy, transfer = MMT.cal_new_transfer_and_last_loss_xy(
                transfer, match_seq, mag_map, BLOCK_SIZE, STEP
            )

    return


if __name__ == '__main__':
    main()
