import argparse
import numpy as np
import math
import copy
import csv
import time
import calendar
import params_experiment.mag_position_with_ilocator_for_experiment as mpp

# 1、读取两个文件作为输入参数：file_list_infcenter.csv（定位） params_factory.csv（每个参数的名称，起点，终点，粒度。复合参数分多行）
# 2、外层循环，构造的参数；内层循环，每个定位文件；
# 3、返回总距离、保存，为了防止保存失败，使用IO流实时写结果

# 一些固定的不需要实验的参数
BLOCK_SIZE = 0.3
DOWN_SIP_DIS = BLOCK_SIZE
SLIDE_BLOCK_SIZE = DOWN_SIP_DIS


# 计算的 滑动距离(SLIDE_STEP * SLIDE_BLOCK_SIZE)要 < 滑动窗口大小(BUFFER_DIS)！
def check_params_valid(params_dict):
    if params_dict['SLIDE_STEP'] * SLIDE_BLOCK_SIZE >= params_dict['BUFFER_DIS']:
        return False

    return True


# 递归构建参数，构造成list[]，list里面包含N个dict
def dfs_params_produce(params_ranges, cur_comb_dict, cur_index, all_comb_dict_list):
    if cur_index == len(params_ranges):
        # 递归出口，将结果组合深拷贝到all_comb_dict中
        if check_params_valid(cur_comb_dict):
            all_comb_dict_list.append(copy.deepcopy(cur_comb_dict))
            # print(len(all_comb_dict_list))
        return

    # 遍历当前参数
    cur_pname = params_ranges[cur_index][0]
    cur_prange = params_ranges[cur_index][1]
    for cur_p in cur_prange:
        cur_comb_dict[cur_pname] = cur_p
        dfs_params_produce(params_ranges, cur_comb_dict, cur_index + 1, all_comb_dict_list)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_list", type=str)  # D:\pythonProjects\MagMapAndPosition\params_experiment\file_list_infcenter.csv
    parser.add_argument("params_factory", type=str)   # D:\pythonProjects\MagMapAndPosition\params_experiment\params_factory.csv
    args = parser.parse_args()
    file_list_path = args.file_list
    params_factory_path = args.params_factory

    # file_list文件结构：第一行是mag_map+move_xy，剩余的是数据对{.npy, _sync.csv}
    # 1、读出地图、文件，作为mag_position_with_ilocator_for_exeriment的输入参数PATH_PDR_RAW, PATH_MAG_MAP
    file_list_ndarr = np.loadtxt(file_list_path, delimiter=',', dtype=str)
    PATH_MAG_MAP = [file_list_ndarr[0][0], file_list_ndarr[0][1]]
    MOVE_X, MOVE_Y = float(file_list_ndarr[0][2]), float(file_list_ndarr[0][3])
    PATH_PDR_RAW_list = []
    for i in range(1, len(file_list_ndarr)):
        PATH_PDR_RAW_list.append([file_list_ndarr[i][0], file_list_ndarr[i][1]])

    # params_factory文件结构[N][4]：参数名，最小值，最大值，变化步长
    # 2、读出参数工厂，根据里面的内容构造出所有的参数的可能，
    params_factory = np.loadtxt(params_factory_path, delimiter=',', dtype=str)
    params_ranges = []  # 每一行存储，[N][参数名, [所有参数值]]
    all_combs = 1
    for line in params_factory:
        temp_list = []
        temp_list.append(line[0])  # str类型参数名
        temp_list.append([])  # 保存所有参数，使用float

        cur_p = float(line[1])
        end_p = float(line[2])
        step = float(line[3])
        while cur_p <= end_p:
            temp_list[1].append(cur_p)
            cur_p += step

        all_combs *= len(temp_list[1])
        params_ranges.append(temp_list)

    # 3、再将params_ranges这些可能组合起来，扔到mag_position_with_ilocator_for_exeriment里面去跑
    #   注意计算的 滑动距离(SLIDE_STEP * SLIDE_BLOCK_SIZE)要 < 滑动窗口大小！需要检查参数合法性的
    all_comb_dict_list = []
    for line in params_ranges:
        print(line)

    print(all_combs)
    dfs_params_produce(params_ranges, {}, 0, all_comb_dict_list)

    print("All params comb has: ", len(all_comb_dict_list))

    # 开启保存参数实验结果的文件
    result_save_dir = "D:\pythonProjects\MagMapAndPosition\params_experiment\\"
    result_csv_file = result_save_dir + str(calendar.timegm(time.gmtime())) + '_params_experiment.csv'
    with open(result_csv_file, "a+", newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 4、遍历所有组合参数
        i = 0
        for params_dict in all_comb_dict_list:
            print(i, '/', len(all_comb_dict_list), ':', params_dict)
            i += 1
            # 遍历所有测试文件
            writer.writerow([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
            csvfile.flush()
            for PATH_PDR_RAW in PATH_PDR_RAW_list:
                # 扔到mag_position_with_ilocator_for_exeriment里面去跑
                cur_result_list = mpp.mag_position_with_ilcator(
                    MOVE_X, MOVE_Y,
                    BLOCK_SIZE,
                    EMD_FILTER_LEVEL=3,
                    BUFFER_DIS=params_dict['BUFFER_DIS'],
                    DOWN_SIP_DIS=DOWN_SIP_DIS,
                    SLIDE_STEP=params_dict['SLIDE_STEP'],
                    SLIDE_BLOCK_SIZE=SLIDE_BLOCK_SIZE,
                    MAX_ITERATION=int(params_dict['MAX_ITERATION']),
                    TARGET_MEAN_LOSS=params_dict['TARGET_MEAN_LOSS'],
                    STEP=params_dict['STEP'],
                    UPPER_LIMIT_OF_GAUSSNEWTEON=500 * params_dict['STEP'] * (params_dict['MAX_ITERATION'] - 1),
                    PDR_IMU_ALIGN_SIZE=10,
                    TRANSFERS_PRODUCE_CONFIG=[
                        [params_dict['TRANSFERS_PRODUCE_CONFIG[0][0]'],
                         params_dict['TRANSFERS_PRODUCE_CONFIG[0][1]'],
                         math.radians(params_dict['TRANSFERS_PRODUCE_CONFIG[0][2]'])]  # 注意读出来的△angle(角度)要加上math.radians(△angle)才能用
                        ,
                        [int(params_dict['TRANSFERS_PRODUCE_CONFIG[1][0]']),
                         int(params_dict['TRANSFERS_PRODUCE_CONFIG[1][1]']),
                         int(params_dict['TRANSFERS_PRODUCE_CONFIG[1][2]'])]
                    ],
                    ORIGINAL_START_TRANSFER=[0., 0., math.radians(0.)],
                    PATH_PDR_RAW=PATH_PDR_RAW,
                    PATH_MAG_MAP=PATH_MAG_MAP
                )

                # 将mag_position_with_ilocator_for_exeriment返回的结果和参数一起存一行excel
                #  每一行是一个实验：前面的列保存使用的参数、后面的列保存误差信息、+运行时间、+总距离、运行时间/总距离;
                #   而且要开一个边跑边写的IO文件流操作，防止跑了半天结果全丢了
                if cur_result_list is not None:
                    cur_result_list.append(params_dict)
                    cur_result_list.append(PATH_PDR_RAW)
                    print(cur_result_list)
                    writer.writerow(cur_result_list)
                    csvfile.flush()

        csvfile.close()