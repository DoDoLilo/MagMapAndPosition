import mag_mapping_tools as MMT


def main():
    path = "data/data_test/data_server_room/IMU-2-3-184.99230319881104 Pixel 6_sync.csv"
    data_all = MMT.get_data_from_csv(path)
    data_mag = data_all[:, 21:24]
    data_g = data_all[:, 24:27]
    data_ori = data_all[:, 18:21]
    # 地磁总强度，垂直、水平分量，
    data_magnitude = MMT.cal_magnitude(data_mag)
    arr_mv_mh = MMT.get_mag_hv_arr(data_ori, data_mag)
    # emd滤波
    mv_filtered_emd = MMT.lowpass_emd(arr_mv_mh[:, 0], 4)
    mh_filtered_emd = MMT.lowpass_emd(arr_mv_mh[:, 1], 4)
    magnitude_filtered_emd = MMT.lowpass_emd(data_magnitude, 4)
    MMT.paint_signal(magnitude_filtered_emd, path)
    MMT.paint_signal(mv_filtered_emd, path + '-mv')
    MMT.paint_signal(mh_filtered_emd, path + '-mh')
    # temp = magnitude_filtered_emd*magnitude_filtered_emd-mv_filtered_emd*mv_filtered_emd-mh_filtered_emd*mh_filtered_emd
    # print(max(temp))
    # print(min(temp))
    # temp = data_magnitude*data_magnitude-arr_mv_mh[:, 0]*arr_mv_mh[:, 0]-arr_mv_mh[:, 1]*arr_mv_mh[:, 1]
    # print(max(temp))
    # print(min(temp))
#    所以是先滤波地磁再算分量，还是算了分量后滤波？


if __name__ == '__main__':
    main()
