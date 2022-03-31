import math

import mag_mapping_tools as MMT
import numpy as np

# 坐标系位移（平移）
MOVE_X = 7.
MOVE_Y = 8.
# 地图坐标系大小 0-MAP_SIZE_X ，0-MAP_SIZE_Y
MAP_SIZE_X = 8.
MAP_SIZE_Y = 13.
# 地图地磁块大小
BLOCK_SIZE = 0.25
# 内插半径
INTER_RADIUS = 1
# 内插迭代次数上限
INTER_TIME_THR = 2
# 删除多余内插块的程度，越大删除的内插范围越大，可以为负值。
DELETE_LEVEL = 0
# ------------------------
# 缓冲池大小，单位（m）
BUFFER_DIS = 10
# 下采样粒度，应为块大小的整数倍
DOWN_SIP_DIS = BLOCK_SIZE
# 高斯牛顿最大迭代次数
MAX_ITERATION = 50
# 目标损失
TARGET_LOSS = BUFFER_DIS/BLOCK_SIZE * 40
# 迭代步长，牛顿高斯迭代是局部最优，步长要小
STEP = 1 / 100000
# 首次迭代固定区域遍历数组，默认起点在某一固定区域
START_TRANSFERS = [[7., 8., math.radians(0.)],
                   [7., 8.5, math.radians(0.)],
                   [7., 7.5, math.radians(0.)],
                   [6.5, 8., math.radians(0.)],
                   [6.5, 8.5, math.radians(0.)],
                   [6.5, 7.5, math.radians(0.)],
                   [7.5, 8., math.radians(0.)],
                   [7.5, 8.5, math.radians(0.)],
                   [7.5, 7.5, math.radians(0.)]]


def main():
    # 全流程
    # 1、建库
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
        delete=True,
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
    #   3.1第一次匹配进行起点周围区域遍历，遍历后选出残差最小的
    #     .取出第一条匹配序列
    # 对匹配序列中的地磁进行滤波，不在这里滤波，在buffer中下采样之前滤波，下采样之后太短了不能滤波？
    first_match = np.array(match_seq_list[0])
    # 此时迭代结束条件是否应该有所不同？目的确实是：找出loss最小的吗？
    # 那么，结束条件应该为相同的迭代次数，而非loss阈值
    # 如果越界，则保存越界前的最后一次轨迹
    candidates_loss_xy_tf = []
    last_loss_xy_tf_num = None
    for tf in START_TRANSFERS:
        for iter_num in range(0, MAX_ITERATION):
            out_of_map, loss, map_xy, tf = MMT.cal_new_transfer_and_last_loss_xy(
                tf, first_match, mag_map, BLOCK_SIZE, STEP
            )
            if not out_of_map:
                last_loss_xy_tf_num = [loss, map_xy, tf, iter_num]
            else:
                print("the out iter num:", iter_num)
                break

        if last_loss_xy_tf_num is not None:
            candidates_loss_xy_tf.append(last_loss_xy_tf_num)

    # 从candidates中挑选出合适的（目前是loss最小and小于Target的）轨迹map_xy与transfer，轨迹用来绘制，transfer进入下次迭代
    transfer = None
    map_xy = None
    min_loss = None
    print("candidates loss:")
    for c in candidates_loss_xy_tf:
        print(c[0])
        if c[0] < TARGET_LOSS:
            if min_loss is None or c[0] < min_loss:
                min_loss = c[0]
                map_xy = c[1]
                transfer = c[2]
    if transfer is None:
        print("初始区域遍历失败，无法找到匹配起始轨迹！")
        return
    print("The Start Track(Loss={0}, Target Loss={2},Transfer={1}):".format(min_loss, transfer, TARGET_LOSS))
    #    3.2后续轨迹基于初始匹配进行迭代------------------------------------------------------------------------------------------
    map_xy_list = [map_xy]
    for i in range(1, len(match_seq_list)):
        match_seq = np.array(match_seq_list[i])
        print("\nMatch Seq:{0}   --------------------------".format(i))
        iter_num = 0
        loss_list = []
        last_transfer = transfer
        while True:
            iter_num += 1
            out_of_map, loss, map_xy, transfer = MMT.cal_new_transfer_and_last_loss_xy(
                transfer, match_seq, mag_map, BLOCK_SIZE, STEP
            )
            if not out_of_map:
                loss_list.append(loss)

            if out_of_map or iter_num > MAX_ITERATION:
                print("Match Seq:", i, "iteration ", iter_num, ", Failed!")
                # 失败了怎么办？提前结束？or 相信PDR不改变transfer？or 使用最小loss的transfer ？
                # 选择：相信PDR和之前的transfer
                # NOTE:失败了还要还原之前的transfer，因为迭代过程中transfer一直在变化！！！
                print("中间轨迹匹配失败！选择：相信PDR和之前的transfer")
                transfer = last_transfer
                map_xy_list.append(MMT.transfer_axis_list(match_seq[:, 0:2], transfer))
                break
            if loss <= TARGET_LOSS:
                print("Match Seq:", i, "iteration ", iter_num, ", Succeed!")
                # 成功了怎么办？加到结果中，提前结束迭代
                map_xy_list.append(map_xy)
                break

        print(loss_list)

    final_xy = []
    for map_xy in map_xy_list:
        for xy in map_xy:
            final_xy.append(xy)
    final_xy = np.array(final_xy)
    MMT.paint_xy(final_xy, "The final xy")
    return


if __name__ == '__main__':
    main()
