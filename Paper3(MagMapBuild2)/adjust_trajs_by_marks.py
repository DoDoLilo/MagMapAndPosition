# TODO
#  1、先计算原始的xy误差，
#  2、手工列出gt xy中哪些点作为特征地标(绘图)，作为adjust_pdr_by_markpoints.py中的marks.csv文件
#  3、按照路遥文献的误差水平，对 random(标准差=0.37米) 随机范围内的 pdrxy，判断是否落于地标内，与地标进行绑定
#    （*注意连续落入地标的pdrxy只取其中均值！否则会出现连续的冗余校准地标），
#       输入到adjust_pdr_by_markpoints.py中进行校准即可
#  再计算校准（合并）后的xy误差
import math

import numpy as np
import paint_tools as PT
import random


def two_points_dis(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 计算两直线斜率，k2相对k1逆时针旋转的度数
def two_slope_angle_off(v1, v2):
    # 方向向量
    x1, y1 = v1
    x2, y2 = v2
    det = x1 * y2 - y1 * x2
    dot = x1 * x2 + y1 * y2
    theta = np.arctan2(det, dot)
    theta = theta if theta > 0 else 2 * np.pi + theta
    return theta


# 坐标变换函数，论文公式4.7。求x1, y1关于transfer的偏导矩阵，论文公式4.13
# 输入：变换向量transfer=[_x, _y, _angle(弧度，绕PDR坐标系的(0, 0)逆时针)]，被变换坐标PDR(x0,y0)
# 输出：转换后的坐标 x1, y1，偏导矩阵 grad_xy(2*3)
# NOTE: t_angle是弧度！不是°。 degrees(x) : 将弧度转化为角度  radians(x) : 将角度转化为弧度。
def transfer_axis(transfer, x0, y0):
    _x = transfer[0]
    _y = transfer[1]
    _angle = transfer[2]
    m_angle = np.array([[math.cos(_angle), -math.sin(_angle)],
                        [math.sin(_angle), math.cos(_angle)]])
    m_move = np.array([[_x],
                       [_y]])
    m_xy = np.array([[x0],
                     [y0]])
    ans = np.dot(m_angle, m_xy) + m_move
    return ans[0][0], ans[1][0]

def mean_dis_bewteen_two_trajs(pdr_xy, gt_xy):
    dis_err = 0
    for pxy, gxy in zip(pdr_xy, gt_xy):
        dis_err += math.sqrt((pxy[0] - gxy[0]) ** 2 + (gxy[1] - gxy[1]) ** 2)
    return dis_err/len(dis_err)


if __name__ == '__main__':
    # 信息中心
    MOVE_X = 5.
    MOVE_Y = 5.
    MAP_SIZE_X = 35.
    MAP_SIZE_Y = 20.

    # 星湖楼
    # MOVE_X = 10.0
    # MOVE_Y = 15.0
    # MAP_SIZE_X = 70.0
    # MAP_SIZE_Y = 28.0

    # 与1相比，最大的缩放比例
    MAX_ZOOM_K_DIS = 0.3
    #
    file = '../Paper3(MagMapBuild2)/results/InfCenter/mag_q_gt_pdr.csv'
    save_file = 'results/InfCenter/new_mag_q_gt_pdr.csv'

    # file = '../Paper3(MagMapBuild2)/results/XingHu/mag_q_gt_pdr.csv'
    # save_file = '../Paper3(MagMapBuild2)/results/XingHu/new_mag_q_gt_pdr.csv'
    print(file)

    mag_q_gt_pdr = np.loadtxt(file, delimiter=',')  # mag 0 1 2, quat 3 4 5 6, gt 7 8, pdr 9 10
    gt_xy = mag_q_gt_pdr[:, 7:9]
    pdr_xy = mag_q_gt_pdr[:, 9:11]

    # 手动设定地标坐标
    XingHu_marks = [  # 星湖楼的地标
        [10, 15],
        [11.7, 17.5], [16.1, 11.5], [18.5, 15.6], [23.3, 17.3], [28.6, 15.2],
        [24.3, 12.1], [26.1, 7.5], [33.7, 18.8], [35.0, 15.9],
        [42.2, 15.4], [45.0, 17.7], [50.2, 14.9], [50.9, 12.6], [53.9, 12.5], [55.0, 15.8],
        [36.6, 14.7], [19.0, 18.1],
    ]

    InfCenter_marks = [
        [5, 5],
        [4.6, 9.2], [8.5, 14.1], [7.3, 5.7], [11.7, 5.1], [12.97, 14.15], [14.66, 9.65],
        [16.9, 12.12], [16.86, 6.25], [21.56, 14.3], [21.8, 8.93], [26.83, 9.05], [26.24, 6.0],
        [15.73, 12.63], [15.6, 8.76]
    ]


    # marks = XingHu_marks
    marks = InfCenter_marks

    for i in range(0, len(gt_xy)):
        if i % 500 == 0:
            marks.append(gt_xy[i].copy())

    marks = np.array(marks)

    # 计算地标平均距离、密度，星湖楼200平方米，信息中心240平方米。
    print("XingHu地标密度：{0}平方米/个地标".format(200 / len(XingHu_marks)))
    print("InfCenter地标密度：{0}平方米/个地标".format(240 / len(InfCenter_marks)))

    # 双重遍历，统计gt_xy 与地标欧氏距离属于 random(0~0.37~0.65)/2 内的数据，进行符合 adjust_pdr_by_markpoints.py 输入格式的绑定
    # *注意连续落入地标的gt_xy只取其中均值！否则会出现连续的冗余校准地标，
    #  通过调整random范围，调整误差高低（进行消融实验）
    marked_index = []  # [N][pdr_index, mx, my]
    for mark in marks:
        mx = mark[0]
        my = mark[1]
        for i in range(0, len(gt_xy)):
            gx = gt_xy[i][0]
            gy = gt_xy[i][1]
            # 给出随机范围
            # half_mark_range = random.uniform(0.1, 0.3) / 2  # uniform(a, b)：生成一个在[a, b]范围内的随机浮点数。
            half_mark_range = 0.1
            # 判断此时gt是否落于该范围
            if math.sqrt((mx - gx) ** 2 + (my - gy) ** 2) <= half_mark_range:
                marked_index.append([i, mx, my, gx, gy])

    # 判断marked_index中的连续index，只保留中间的index
    marked_index = np.array(marked_index)
    new_marked_index = []
    start_i = 0
    print(len(marked_index))
    print(marked_index)

    while start_i < len(marked_index):
        if start_i + 1 >= len(marked_index):
            break
        for end_i in range(start_i + 1, len(marked_index)):
            if marked_index[end_i][0] - marked_index[end_i - 1][0] > 3 or end_i == len(marked_index) - 1:
                # 不是连续的or来到最后还是连续的，则将中间结果存入新的
                new_marked_index.append(marked_index[int((start_i + end_i) / 2)].copy())
                start_i = end_i
                break
    new_marked_index = np.array(new_marked_index)  # [0:index, 1:mx, 2:my, 3:gx, 4:gy]

    # 使用argsort()函数获取第一列的排序索引
    sorted_indices = np.argsort(new_marked_index[:, 0])
    # 使用索引数组对整个数组进行排序
    sorted_marked_index = new_marked_index[sorted_indices]

    print(len(new_marked_index))
    print(new_marked_index)

    # 绘制 mark_xy 还有 marked_gt_xy
    # PT.paint_xy_list([new_marked_index[:, 1:3], new_marked_index[:, 3:5]],
    #                  ['mark_xy', 'marked_gt_xy'],
    #                  [0, 70 * 1, 0, 28 * 1], '')

    # TODO 根据矫正点，对pdr xy 进行 adjust_pdr_by_markpoints.py中的 校准
    # 过滤缩放比例k较大的轨迹段，这些可能是有bug的
    # 根据排序后的下标，根据相邻标记点，进行分段，TODO 舍弃首尾无地标包围的段落
    # sorted_marked_index[0:index, 1:mx, 2:my, 3:gx, 4:gy]
    new_mag_q_gt_pdr = []  # mag 0 1 2, quat 3 4 5 6, gt 7 8, pdr 9 10
    start_transfer = None
    for i in range(1, len(sorted_marked_index)):
        # pdr子段
        start_i = sorted_marked_index[i - 1][0]
        end_i = sorted_marked_index[i][0]
        sub_pdr_xy = pdr_xy[int(start_i): int(end_i + 1), :].copy()
        sub_gt_xy = gt_xy[int(start_i): int(end_i + 1), :].copy()
        # pdr始末点
        start_pdr_xy = sub_pdr_xy[0].copy()
        end_pdr_xy = sub_pdr_xy[-1].copy()

        # start_mark_xy = sorted_marked_index[i - 1, 1:3].copy()
        # end_mark_xy = sorted_marked_index[i, 1:3].copy()

        start_mark_xy = sub_gt_xy[0].copy()
        end_mark_xy = sub_gt_xy[-1].copy()

        # 缩放比例
        zoom_k = two_points_dis(start_mark_xy, end_mark_xy) / two_points_dis(start_pdr_xy, end_pdr_xy) if two_points_dis(start_pdr_xy, end_pdr_xy) !=0 else 0
        if abs(1 - zoom_k) > MAX_ZOOM_K_DIS:
            continue
        print("zoom_k:", zoom_k)
        # 将该段pdr所有的相邻坐标都进行缩放，不会改变始末点直线的斜率
        for j in range(1, len(sub_pdr_xy)):
            # 平移到原点
            move_x = -sub_pdr_xy[j - 1, 0]
            move_y = -sub_pdr_xy[j - 1, 1]
            sub_pdr_xy[:, 0] += move_x
            sub_pdr_xy[:, 1] += move_y
            # 缩放距离
            old_xy = sub_pdr_xy[j]
            new_xy = old_xy * zoom_k
            # 将 j 及其之后的所有点平移
            move_x_1 = new_xy[0] - old_xy[0]
            move_y_1 = new_xy[1] - old_xy[1]
            sub_pdr_xy[j:, 0] += move_x_1
            sub_pdr_xy[j:, 1] += move_y_1

        # 将这段放大后的pdr旋转至与mark_xy对齐
        # 计算两直线的 始末点向量 夹角，方向向量 = end_point - start_point
        vector_pdr = end_pdr_xy[0] - start_pdr_xy[0], end_pdr_xy[1] - start_pdr_xy[1]
        vector_mark = end_mark_xy[0] - start_mark_xy[0], end_mark_xy[1] - start_mark_xy[1]
        angle_off = two_slope_angle_off(vector_pdr, vector_mark)
        # 坐标转换
        # 先平移到原点，再计算变换向量（因为是先绕原点逆时针旋转再平移）
        move_x_2 = sub_pdr_xy[0, 0]
        move_y_2 = sub_pdr_xy[0, 1]
        sub_pdr_xy[:, 0] -= move_x_2
        sub_pdr_xy[:, 1] -= move_y_2
        transfer = [start_mark_xy[0], start_mark_xy[1], angle_off]
        start_transfer = transfer.copy() if i == 1 else start_transfer
        print("transfer", transfer[0], ',', transfer[1], ',', math.degrees(transfer[2]), '\n')
        new_sub_pdr_xy = []
        offset = 0
        for xy in sub_pdr_xy:
            x, y = transfer_axis(transfer, xy[0], xy[1])

            temp = mag_q_gt_pdr[int(start_i + offset)]  # mag 0 1 2, quat 3 4 5 6, gt 7 8, pdr 9 10
            new_mag_q_gt_pdr.append([
                temp[0], temp[1], temp[2],
                temp[3], temp[4], temp[5], temp[6],
                temp[7], temp[8],
                x, y
            ])
            offset += 1

        print('Len sub_pdr_xy', len(sub_pdr_xy), 'Len new sub_pdr_xy', len(new_sub_pdr_xy))

    # 绑定数据，计算与gt xy 的误差水平
    new_mag_q_gt_pdr = np.array(new_mag_q_gt_pdr)
    map_size = [0, MAP_SIZE_X * 1, 0, MAP_SIZE_Y * 1]
    PT.paint_xy_list([mag_q_gt_pdr[:, 7:9]], ['GT'], map_size, '')
    PT.paint_xy_list([mag_q_gt_pdr[:, 9:11]], ['PDR'], map_size, '')
    PT.paint_xy_list([mag_q_gt_pdr[:, 7:9], mag_q_gt_pdr[:, 9:11]], ['PDR', 'GT'], map_size, '')
    # 统计轨迹距离dis，和GT误差
    dis_err = 0
    for pxy, gxy in zip(pdr_xy, gt_xy):
        dis_err += math.sqrt((pxy[0] - gxy[0]) ** 2 + (gxy[1] - gxy[1]) ** 2)
    print("Raw xy mean error = ", dis_err / len(gt_xy))

    PT.paint_xy_list([new_mag_q_gt_pdr[:, 7:9]], ['GT'], map_size, '')
    PT.paint_xy_list([new_mag_q_gt_pdr[:, 9:11]], ['PDR'], map_size, '')
    PT.paint_xy_list([new_mag_q_gt_pdr[:, 7:9], new_mag_q_gt_pdr[:, 9:11]], ['PDR', 'GT'], map_size, '')

    # 统计轨迹距离dis，和GT误差
    dis_err = 0
    for pxy, gxy in zip(new_mag_q_gt_pdr[:, 7:9], new_mag_q_gt_pdr[:, 9:11]):
        dis_err += math.sqrt((pxy[0] - gxy[0]) ** 2 + (gxy[1] - gxy[1]) ** 2)
    print("New xy mean error = ", dis_err / len(gt_xy))

    # TODO 保存结果new_mag_q_gt_pdr
    np.savetxt(save_file, new_mag_q_gt_pdr, delimiter=',')
