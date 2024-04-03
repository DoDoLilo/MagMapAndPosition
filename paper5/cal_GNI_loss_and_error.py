# 5.1.2节实验结果
# 读取所有文件，合并到一起 = [N][Bad Loss Percent, Win Mean Error]
# 统计最大和最小BLP，将其分割为几个区间，统计区间占比（CDF）
# 统计对应的WME大小

import numpy as np

files = [
    "../paper5/results/badLoss_and_Winerror(Inited_XingHu).csv",
    "../paper5/results/badLoss_and_Winerror(InReal_XingHu).csv",
    # "../paper5/results/badLoss_and_Winerror(Inited_InfCenter).csv",  # 这个结果很垃圾
    # "../paper5/results/badLoss_and_Winerror(InReal_InfCenter).csv" # 这个结果很垃圾
]

if __name__ == '__main__':
    all_BLP_WME = []
    for file in files:
        data = np.loadtxt(file, delimiter=',')
        for d in data:
            all_BLP_WME.append([d[0], d[1]])

    all_BLP_WME = np.array(all_BLP_WME)
    #
    max_BLP = max(all_BLP_WME[:, 0])
    min_BLP = 0

    print(max_BLP, ',', min_BLP)
    # 分割子区间 2% 一个
    sub_win = 0.02

    sub_list = []
    for i in range(0, int(1/sub_win)):
        sub_list.append([])

    for d in all_BLP_WME:
        blp = d[0]
        wme = d[1]
        sub_list[int(blp/sub_win)].append(wme)

    # 计算区间误差均值，保存结果=[N][占比，误差]
    result = np.empty((int(1/sub_win), 2))

    all_num = len(all_BLP_WME)
    for i in range(0, len(sub_list)):
        result[i][0] = len(sub_list[i])/all_num
        # result[i][1] = sum(sub_list[i])/len(sub_list[i]) if len(sub_list[i]) != 0 else 0
        result[i][1] = max(sub_list[i]) if len(sub_list[i]) != 0 else 0

    np.savetxt("../paper5/results/BLPP_MWME.csv", result, delimiter=',')



