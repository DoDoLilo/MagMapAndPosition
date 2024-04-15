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
    return dis_err/len(gt_xy)


if __name__ == '__main__':
    # 信息中心
    # MOVE_X = 5.
    # MOVE_Y = 5.
    # MAP_SIZE_X = 35.
    # MAP_SIZE_Y = 20.

    # 星湖楼
    MOVE_X = 10.0
    MOVE_Y = 15.0
    MAP_SIZE_X = 70.0
    MAP_SIZE_Y = 28.0

    # 与1相比，最大的缩放比例
    MAX_ZOOM_K_DIS = 0.1
    #
    # file = '../Paper3(MagMapBuild2)/results/InfCenter/mag_q_gt_pdr.csv'
    # save_file = 'results/InfCenter/new_mag_q_gt_pdr.csv'

    file = '../Paper3(MagMapBuild2)/results/XingHu/mag_q_gt_pdr.csv'
    save_file = '../Paper3(MagMapBuild2)/results/XingHu/new_mag_q_gt_pdr.csv'
    print(file)

    mag_q_gt_pdr = np.loadtxt(file, delimiter=',')  # mag 0 1 2, quat 3 4 5 6, gt 7 8, pdr 9 10
    gt_xy = mag_q_gt_pdr[:, 7:9]
    pdr_xy = mag_q_gt_pdr[:, 9:11]

    new_marked_index = []
    for i in range(0, len(gt_xy)):
        if i % 300 == 0:
            new_marked_index.append([i, gt_xy[i][0], gt_xy[i][1]])

    new_marked_index = np.array(new_marked_index)  # [0:index, 1:mx, 2:my, 3:gx, 4:gy]

    print(len(new_marked_index))
    print(new_marked_index)

    # 根据矫正点，对pdr xy 进行 adjust_pdr_by_markpoints.py中的 校准
    # 过滤缩放比例k较大的轨迹段，这些可能是有bug的
    # 根据排序后的下标，根据相邻标记点，进行分段， 舍弃首尾无地标包围的段落
    # sorted_marked_index[0:index, 1:mx, 2:my, 3:gx, 4:gy]
    new_mag_q_gt_pdr = []  # mag 0 1 2, quat 3 4 5 6, gt 7 8, pdr 9 10
    start_transfer = None
    for i in range(1, len(new_marked_index)):
        # pdr子段
        start_i = new_marked_index[i - 1][0]
        end_i = new_marked_index[i][0]
        sub_pdr_xy = pdr_xy[int(start_i): int(end_i + 1), :].copy()
        sub_gt_xy = gt_xy[int(start_i): int(end_i + 1), :].copy()
        # pdr始末点
        start_pdr_xy = sub_pdr_xy[0].copy()
        end_pdr_xy = sub_pdr_xy[-1].copy()

        start_mark_xy = new_marked_index[i - 1, 1:3].copy() + random.uniform(0.1, 0.3)
        end_mark_xy = new_marked_index[i, 1:3].copy() + random.uniform(0.1, 0.3)

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
    print("New xy mean error = ", mean_dis_bewteen_two_trajs(new_mag_q_gt_pdr[:, 7:9], new_mag_q_gt_pdr[:, 9:11]))

    # TODO 保存结果new_mag_q_gt_pdr
    np.savetxt(save_file, new_mag_q_gt_pdr, delimiter=',')
