import numpy as np
import statistics

# 移除离群值：
# 读取MFR与RMSD
file = r'../paper_4/results/All_features_RMSD_all_initByGT(paper).csv'

# 按照指定MFR间隔统计子区间离散数量# seq_fc, seq_td, area_fc, area_td
# result = np.loadtxt(file, delimiter=',')[:, 0:2]
# result = np.loadtxt(file, delimiter=',')[:, 2:4]
# result = np.loadtxt(file, delimiter=',')[:, 4:6]
result = np.loadtxt(file, delimiter=',')[:, 6:8]
mfr = result[0:228, 0]
rmsd = result[0:228, 1]

step = 0.1
removed_js = []
k = 2
for i in range(0, int(1/step)):
    s = i * step
    e = (i+1) * step
    temp_rmsd = []
    temp_j = []
    for j in range(0, len(mfr)):
        if s <= mfr[j] < e:
            temp_rmsd.append(rmsd[j])
            temp_j.append(j)
    # 统计rmsd，记录要删除的j
    # 计算标准差、均值
    if len(temp_rmsd) < 2:
        continue
    std_dev = statistics.stdev(temp_rmsd)
    mean_rmsd = sum(temp_rmsd)/len(temp_rmsd)
    for j2 in range(0, len(temp_rmsd)):
        if temp_rmsd[j2] < (mean_rmsd-k*std_dev) or temp_rmsd[j2] > (mean_rmsd+k*std_dev):
            removed_js.append(temp_j[j2])

print(removed_js)
print(len(removed_js)/len(mfr))

new_result = []
for i in range(0, len(result)):
    if i not in removed_js:
        new_result.append(result[i, :])

new_result = np.array(new_result)
print(len(new_result))
# seq_fc, seq_td, area_fc, area_td
np.savetxt('../paper_4/results/area_td_filtered.csv', new_result, delimiter=',')