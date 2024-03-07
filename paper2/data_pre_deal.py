# 提取出MMT中的低通滤波、去均值
import numpy as np
import mag_mapping_tools as MMT
import math
import pandas

# files = ["data/IMU-1029-1-83.49090052734772 Pixel 6.csv",
#          "data/IMU-1029-1-86.60236588299034 M2011K2C.csv"]
files = ["data/IMU-1029-2-93.94119924049394 Pixel 6.csv",
         "data/IMU-1029-2-94.69853061566496 Pixel 3a.csv",
         "data/IMU-1029-2-128.77568993212506 M2011K2C.csv"]
# for 循环读取文件
all_mag = []
all_time = []
for f in files:
    all_time.append(MMT.get_data_from_csv(f)[:, 0])
    temp_mag = []
    for m in MMT.get_data_from_csv(f)[:, 21:24]:
        temp_mag.append(math.sqrt(m[0]**2+m[1]**2+m[2]**2))
    all_mag.append(temp_mag)

# 每个文件重置时间戳，归一化到起点为0
for time in all_time:
    start_time = time[0]
    for i in range(0, len(time)):
        time[i] -= start_time

# 找到按照最短的文件
min_index = 0
min_len = len(all_time[0])
for i in range(1, len(all_time)):
    if len(all_time[i]) < min_len:
        min_len = len(all_time[i])
        min_index = i

# 进行时间对齐
align_all_mag = []
align_all_mag.append(all_mag[min_index])
print(min_index)
for i in range(0, len(all_time)):
    if i == min_index:
        continue
    else:
        # 时间对齐
        temp_list = []
        j0 = 0
        for j in range(0, len(all_time[i])):
            t0 = all_time[min_index][j0]
            t = all_time[i][j]
            m = all_mag[i][j]
            if t >= t0:
                temp_list.append(m)
                j0 += 1
                if j0 >= len(all_time[min_index]):
                    break
        align_all_mag.append(temp_list)

# 只截取对齐后的中间一段
for i in range(0, len(align_all_mag)):
    align_all_mag[i] = align_all_mag[i][900:2900]

ni = 0
for align_mag in align_all_mag:
    np.savetxt("data/{0}_raw_mags.csv".format(ni), align_mag, delimiter=',')
    ni += 1

# 低通滤波
all_filtered_mag = []
for mag_x in align_all_mag:
    all_filtered_mag.append(MMT.lowpass_emd(np.array(mag_x), cut_off=4))
ni = 0
for filtered_mag in all_filtered_mag:
    np.savetxt("data/{0}_filtered_mags.csv".format(ni), filtered_mag, delimiter=',')
    ni += 1

# 去均值
all_move_mean = all_filtered_mag.copy()
for mag_x in all_move_mean:
    mean = sum(mag_x)/len(mag_x)
    for i in range(0, len(mag_x)):
        mag_x[i] -= mean
ni = 0
for move_mean in all_move_mean:
    np.savetxt("data/{0}_move_mean_mags.csv".format(ni), move_mean, delimiter=',')
    ni += 1
