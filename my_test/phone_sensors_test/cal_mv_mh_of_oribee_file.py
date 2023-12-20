# 用和地磁定位算法一样的方法计算oribeeV2采集到的文件的mv mh
# 结果保存为 [time, game_rotation, mag_filed, mv, mh]


import mag_mapping_tools as MMT
import numpy as np
import os


# get all the files of the path
def get_Listfiles(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for file in files:
            # include path
            Filelist.append(os.path.join(home, file))
            #Filelist.append(file)

    return Filelist

if __name__ == '__main__':
    # oribee_file_list = [
    #     # "D:\pythonProjects\MagMapAndPosition\my_test\phone_sensors_test\IMU-523-1-33.74545800960761 SEA-AL10.csv",
    #     # "D:\pythonProjects\MagMapAndPosition\my_test\phone_sensors_test\IMU-523-1-39.289382540028974 Pixel 6.csv",
    #     # "D:\pythonProjects\MagMapAndPosition\my_test\phone_sensors_test\（平端）IMU-525-1-88.41923718334876 Pixel 3a.csv",
    #     # "D:\pythonProjects\MagMapAndPosition\my_test\phone_sensors_test\（屏幕斜朝自己）IMU-525-2-91.49990521602379 Pixel 3a.csv",
    #     # "D:\pythonProjects\MagMapAndPosition\my_test\phone_sensors_test\（屏幕斜朝左）IMU-525-4-89.3504651747883 Pixel 3a.csv",
    #     # "D:\pythonProjects\MagMapAndPosition\my_test\phone_sensors_test\（屏幕斜朝右）IMU-525-3-87.04416983926345 Pixel 3a.csv",
    #     "D:\pythonProjects\MagMapAndPosition\my_test\phone_sensors_test\（平端-5cm晃）IMU-525-5-90.91684611752567 Pixel 3a.csv",
    #     "D:\pythonProjects\MagMapAndPosition\my_test\phone_sensors_test\（朝自己-5cm晃）IMU-525-6-91.92595875607086 Pixel 3a.csv",
    #     "D:\pythonProjects\MagMapAndPosition\my_test\phone_sensors_test\（朝左-5cm晃）IMU-525-8-90.86949924619913 Pixel 3a.csv",
    #     "D:\pythonProjects\MagMapAndPosition\my_test\phone_sensors_test\（朝右-5cm晃）IMU-525-7-86.18360698497686 Pixel 3a.csv"
    # ]

    oribee_file_list = get_Listfiles(r"C:\Users\14799\OneDrive - whu.edu.cn\桌面\智慧机房测试\实验与测试数据\5.26")
    print(oribee_file_list)

    for oribee_file in oribee_file_list:
        data_all = np.loadtxt(oribee_file, delimiter=',')

        time = data_all[10:, 0]
        game_rotation = data_all[10:, 7:11]
        mag_filed = data_all[10:, 21:24]

        # 计算mv, mh
        arr_mv_mh = MMT.get_2d_mag_qiu(game_rotation, mag_filed)

        # emd滤波
        mv_filtered_emd = MMT.lowpass_emd(arr_mv_mh[:, 0], 4)
        mh_filtered_emd = MMT.lowpass_emd(arr_mv_mh[:, 1], 4)
        arr_mv_mh = np.vstack((mv_filtered_emd, mh_filtered_emd)).transpose()

        # 保存到文件中
        data_to_save = []
        for i in range(0, len(time)):
            data_to_save.append([time[i],
                                 game_rotation[i][0], game_rotation[i][1], game_rotation[i][2], game_rotation[i][3],
                                 mag_filed[i][0], mag_filed[i][1], mag_filed[i][2],
                                 arr_mv_mh[i][0], arr_mv_mh[i][1]])

        save_file_path = os.path.dirname(oribee_file) + "/" + "mvh_" + os.path.basename(oribee_file)
        np.savetxt(save_file_path, data_to_save, delimiter=',', fmt="%.4f")
        print("Save file to:", save_file_path)
