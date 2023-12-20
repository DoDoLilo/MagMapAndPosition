import numpy as np
import csv

# 读取实验结果文件，每group_num一组
# 每列含义是“pdr和gt平均误差，地磁和gt平均误差，一米内的占比，一米内的占比/0.65	耗费时间，耗费时间/米，总轨迹长度”
# 手工设定某几列的筛选条件，对总的结果进行筛选
# 筛选条件要求同一组文件全部满足要求，才保留结果

def data_filter(data_list, col_filt, filt_stand, group_num):
    """
    对输入的实验结果列表按指定的列进行过滤，只留一组内所有数据都满足该列过滤要求的组结果。
    否则要将该组的所有结果+前面的一行分隔行 总共group_num+1行数据删除
    :param data_list:
    :param col_filt:
    :param filt_stand:
    :param group_num:
    :return:
    """

    # 遍历list，步长为group_num+1
    row_i = 0
    while row_i < len(data_list):
        # 跳过第一行的间隔符，判断改组的所有结果是否都满足指定列的过滤条件
        # print("new len", len(data_list), " row_i:", row_i)
        all_in_stand = True
        for gi in range(row_i + 1, row_i + group_num + 1):
            if float(data_list[gi][col_filt]) > filt_stand:
                all_in_stand = False
                break

        if not all_in_stand:
            # 如果存在不满足的，则将该组的所有结果+前面的一行分隔行 总共group_num+1行数据删除
            # 为了避免fast-fail问题，执行删除后 row_i 不增加 group_num+1
            del data_list[row_i: row_i + group_num + 1]
        else:
            row_i += (group_num + 1)

    return


if __name__ == '__main__':
    # 一个参数对应的一组文件的数量
    group_num = 5
    # 要进行过滤的文件
    result_file = "D:\\pythonProjects\\MagMapAndPosition\\params_experiment\\1684128760_params_experiment.csv"
    #
    filtered_result_file = result_file[0:len(result_file) - len(".csv")] + "_filtered" + ".csv"

    # 读取文件，因为列数不一致，需要用csv读取
    result_data_list = []
    f = open(result_file, 'r')
    content = f.read()
    rows = content.split('\n')
    for row in rows:
        result_data_list.append(row.split(','))

    # print(result_data_list)

    # 要过滤的条件，[N][列，过滤条件]
    before_filter = len(result_data_list) / (group_num + 1)
    # filter_conditions = [[1, 0.8], [5, 1]]
    filter_conditions = [[1, 0.8], [5, 0.3]]
    # filter_conditions = [[1, 1]]
    for fc in filter_conditions:
        data_filter(result_data_list, fc[0], fc[1], group_num)

    after_filter = len(result_data_list) / (group_num + 1)

    # 将过滤后的数据保存为结果
    with open(filtered_result_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow("pdr和gt平均误差，地磁和gt平均误差，一米内的占比，一米内的占比/0.65，耗费时间，耗费时间/米，总轨迹长度".split("，"))
        writer.writerows(result_data_list)

    print("How many groups filtered:  ", before_filter - after_filter , '/', before_filter)
    print("save filtered result to: ", filtered_result_file)
