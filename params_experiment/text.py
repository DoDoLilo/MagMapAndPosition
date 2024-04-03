import calendar
import csv
import time

import numpy as np

# file_list_path = "D:\pythonProjects\MagMapAndPosition\params_experiment\\file_list_xinhu.csv"
# data = np.loadtxt(file_list_path, delimiter=',', dtype=str)
#
# for line in data:
#     print(line)

if __name__ == '__main__':
    result_save_dir = "D:\pythonProjects\MagMapAndPosition\params_experiment\\"
    result_csv_file = result_save_dir + str(calendar.timegm(time.gmtime())) + '_params_experiment.csv'
    with open(result_csv_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)

        result_list = [1, 2.2, 3]
        params_dict = {"p1": 1, "p2": 3}
        result_list.append(params_dict)

        writer.writerow(result_list)

        csvfile.close()