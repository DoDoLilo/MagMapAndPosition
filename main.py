import mag_mapping_tools as MMT


def main():
    # 全流程
    # 1、建库（未滤波）

    # 2、缓冲池给匹配段（内置稀疏采样）

    # 3、手动给出初始transfer_0
    # 根据匹配段进行迭代，3种迭代结束情况：
    #  A：迭代out_of_map返回True；B：迭代次数超出阈值但last_loss仍未达标；C：迭代last_loss小于阈值
    # 情况AB表示匹配失败，论文未说匹配失败的解决方法

    return


if __name__ == '__main__':
    main()
