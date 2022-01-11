import mag_mapping_tools as MMT


def main():
    data_all = MMT.get_data_from_csv("data/data_test/机房数据/IMU-1-1-191.0820588816594 Pixel 3a_sync.csv", 0, 29)
    data_mag = data_all[:, 21:24]
    data_g = data_all[:, 24:27]
    data_ori = data_all[:, 18:21]
    arr_mv_mh = MMT.get_mag_hv_arr(data_ori, data_mag)
    print(arr_mv_mh[0, :])

    # data_magnitude = MCT.cal_magnitude(data_mag)
    # data_filtered_butter = MCT.lowpass_butter(data_magnitude, 200, 12)
    # data_filtered_EMD = MCT.lowpass_emd(data_magnitude, 3)
    # data_down_sampling = MCT.down_sampling_by_mean(data_magnitude, 40)


if __name__ == '__main__':
    main()
