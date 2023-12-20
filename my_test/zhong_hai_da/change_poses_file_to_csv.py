# 他们给的poses.csv文件，没有用 "," 作为间隔符，是非标准的csv格式，导致所有数据存在同一列
# 所以需要读取poses.csv文件的第一列的str，通过 " " 分隔得到不同列，再保存为标准csv格式
import os
import numpy as np

if __name__ == '__main__':
    old_pose_file = "D:\pythonProjects\MagMapAndPosition\my_test\zhong_hai_da\pose_no_offset.csv"
    new_pose_file = os.path.dirname(old_pose_file) + "/" + "standard_" + os.path.basename(old_pose_file)

    # 读取旧文件，只有一列
    old_data = np.loadtxt(old_pose_file, dtype=str)
    new_data_list = []
    for line in old_data:
        # 以空格为间隔，type(line) = numpy.ndarray, type(line[0]) = numpy.str
        temp_list = []
        for d_str in line:
            temp_list.append(float(d_str))
        new_data_list.append(temp_list)

    np.savetxt(new_pose_file, new_data_list, delimiter=',', fmt='%.08f')
    print("Save file to: ", new_pose_file)