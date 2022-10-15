import csv
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp
import scipy.io as sio

from utils import beats_show, read_mat


def get_beat_from_recording(recording, isShow=False, isPrintSize=False):
    # 消噪
    recording = wave_filtering_high(recording)
    working_data = {}
    try:
        # 将原始数据放缩到[0:1024]，放大信号
        data = hp.scale_data(recording)
        # 自动获取R波位置
        working_data, measures = hp.process(data, 500.0, interp_clipping=True, interp_threshold=1023)
    except:
        # 这里考虑峰值朝下的情况
        data = hp.scale_data(-recording)
        try:
            working_data, measures = hp.process(data, 500.0, interp_clipping=True, interp_threshold=1023)
        except:
            working_data['peaklist'] = []
            tmp_num = int(len(recording) / 500)
            for i in range(tmp_num):
                working_data['peaklist'].append(i * 500 + 250)

    # 获取R波对应的下标
    peaklists = working_data['peaklist']
    if len(peaklists) - 2 <= 0:
        return []

    # 确定一个beat的长度
    beat_length_average = int((peaklists[-1] - peaklists[0]) / (len(peaklists) - 1))
    beat_left = int(beat_length_average / 2)
    beat_right = beat_length_average - beat_left

    # 去头去尾
    peaklists = peaklists[1:-1]

    # 获取心拍
    beat_data = []
    # 记录每个beat截取时的下标
    beat_split_index = []
    # 记录将原始beat crop到500时左右两边填充/裁剪的距离
    beat_padding_index = []
    beat_list_length = []
    # 记录原始ECG裁剪的左右下标
    ECG_ll = 9999
    ECG_rr = 0

    last_beat_rr = 0
    for i in range(len(peaklists)):
        R_index = peaklists[i]
        current_beat_ll = R_index - beat_left
        current_beat_rr = R_index + beat_right
        if i == 0:
            ECG_ll = current_beat_ll
        if i == len(peaklists) - 1:
            ECG_rr = current_beat_rr

        if last_beat_rr != current_beat_ll and last_beat_rr != 0:
            current_beat_ll = last_beat_rr

        last_beat_rr = current_beat_rr
        beat_list_length.append(current_beat_rr - current_beat_ll)
        tmp_data = recording[current_beat_ll:current_beat_rr]
        tmp_data, padding_item = beat_crop(tmp_data)

        beat_split_index.append([current_beat_ll, current_beat_rr])
        beat_padding_index.append(padding_item)
        beat_data.append(tmp_data)

    if isPrintSize:
        print("ECG leagth:{}".format(len(recording)))
        print("beat avg length:{}, ECG_ll:{}, ECG_rr:{}".format(beat_length_average, ECG_ll, ECG_rr))
        print(len(peaklists), peaklists)
        print("beats {}: {}".format(len(beat_split_index), beat_split_index))
        print(beat_list_length)

    if isShow:
        beats_show(beat_data)

    return beat_list_length, beat_split_index, beat_padding_index, ECG_ll, ECG_rr


def get_beat_from_recording_by_index(recording, beat_split_index, isShow=False):
    # 根据输入的第一导联beat下标进行其他导联beat的切割
    recording = wave_filtering_high(recording)
    # 获取心拍
    beat_data = []
    for item in beat_split_index:
        tmp_data = recording[item[0]:item[1]]

        if isShow:
            beats_show([tmp_data], col=1, row=1)

        tmp_data, _ = beat_crop(tmp_data)

        beat_data.append(tmp_data)

    if isShow:
        beats_show(beat_data)

    return beat_data


def get_index_from_recording_12_lead(ECG_list, noise_leadID_list=None):
    # 提取ECG_list的beat切割下标
    best_lead_id = -1
    beat_split_index = []
    beat_padding_index = []
    ECG_ll, ECG_rr = 0, 0
    for i in range(12):
        # print("test lead:", i)
        if noise_leadID_list is not None and (i + 1) in noise_leadID_list:
            # ID为i+1的导联为noise lead，要跳过
            continue
        beat_list_length, beat_split_index, beat_padding_index, ECG_ll, ECG_rr = get_beat_from_recording(
            ECG_list[i], isShow=False,
            isPrintSize=False)
        if check_beat_index(beat_list_length):
            # 如果当前导联切割的下标不合理，则用下一导联的数据
            best_lead_id = i
            break
        else:
            if best_lead_id == -1 and i == 11:
                if noise_leadID_list is None:
                    beat_list_length, beat_split_index, beat_padding_index, ECG_ll, ECG_rr = get_beat_from_recording(
                        ECG_list[0], isShow=False,
                        isPrintSize=False)
                else:
                    for tmp in range(12):
                        if (tmp + 1) not in noise_leadID_list:
                            beat_list_length, beat_split_index, beat_padding_index, ECG_ll, ECG_rr = get_beat_from_recording(
                                ECG_list[tmp], isShow=False,
                                isPrintSize=False)

    # print("best_lead_id:　{}".format(best_lead_id))

    return beat_split_index, beat_padding_index, ECG_ll, ECG_rr


def check_beat_index(beat_list_length):
    # 检测切割的beat下标是否合理
    beat_length_min = min(beat_list_length)
    beat_length_max = max(beat_list_length)
    # print("Difference:{}, max length:{}, min length:{}".format(beat_length_max - beat_length_min, beat_length_max, beat_length_min))
    # 检查裁剪的每个beat length都在合理范围内
    for item in beat_list_length:
        if item < 100 or item > 700:
            return False

    if beat_length_max - beat_length_min > 150:
        return False
    else:
        return True


def beat_crop(beat_data):
    if len(beat_data) == 0:
        return beat_data, [250, 250]
    if len(beat_data) < 500:
        # 补全至统一长度:500
        ll = int((500 - len(beat_data)) / 2)
        rr = (500 - len(beat_data)) - ll
        for i in range(ll):
            beat_data = np.insert(beat_data, 0, beat_data[0])
        for i in range(rr):
            beat_data = np.insert(beat_data, -1, beat_data[-1])
    else:
        center_index = int(len(beat_data) / 2)
        ll = -(center_index - 250)
        rr = -(len(beat_data) + ll - 500)
        beat_data = beat_data[center_index - 250:center_index + 250]
    # ll,rr记录当前beat左右被裁剪的长度
    return beat_data, [ll, rr]


def ECG_crop(ECG_data, avg_length=5000):
    try:
        for i in range(12):
            ECG_data[i] = ECG_data[i].numpy()
    except:
        pass
    ECG_data = np.array(ECG_data)
    avg_length_half = int(avg_length / 2)

    try:
        ori_length = ECG_data.shape[1]
    except:
        ori_length = ECG_data[0].shape[0]

    if ori_length == avg_length:
        return ECG_data
    elif ori_length < avg_length:
        new_ECG_data = None
        ll = int((avg_length - ori_length) / 2)
        rr = avg_length - ori_length - ll
        for lead_index in range(0, 12):
            tmp_ECG_lead = ECG_data[lead_index]
            for i in range(ll):
                tmp_ECG_lead = np.insert(tmp_ECG_lead, 0, tmp_ECG_lead[0])
            for i in range(rr):
                tmp_ECG_lead = np.insert(tmp_ECG_lead, -1, tmp_ECG_lead[-1])

            if new_ECG_data is None:
                new_ECG_data = tmp_ECG_lead
            else:
                new_ECG_data = np.vstack((new_ECG_data, tmp_ECG_lead))
    else:
        center_index = int(ori_length / 2)
        new_ECG_data = None
        for lead_index in range(0, 12):

            if new_ECG_data is None:
                new_ECG_data = ECG_data[lead_index][center_index - avg_length_half: center_index + avg_length_half]
            else:
                new_ECG_data = np.vstack((new_ECG_data,
                                          ECG_data[lead_index][center_index - avg_length_half: center_index + avg_length_half]))

    return new_ECG_data


# 滤波器
def wave_filtering_high(raw_recording, filter='band'):
    if filter == 'high':
        # 高通滤波 >0.75hz
        data_filter = hp.filter_signal(raw_recording, cutoff=0.75, sample_rate=500.0, order=3, filtertype='highpass')
    elif filter == 'low':
        # 低通滤波 <15hz
        data_filter = hp.filter_signal(raw_recording, cutoff=15, sample_rate=500.0, order=3, filtertype='lowpass')
    else:
        # 带通滤波 [0.75, 15]
        data_filter = hp.filter_signal(raw_recording, cutoff=[0.45, 40], sample_rate=500.0, order=3,
                                       filtertype='bandpass')

    return data_filter


def get_Beat_Generate_dataset_Extract_beat_and_lead_label_from_npdata(dataset_folder):
    # 构造Beat生成模型训练的数据集
    # 对整理好的npdata（ECG train、val、test）读取，进行指定导联beat提取，和 Y label的构造
    # 完成对比实验，控制训练集完全一致
    if not os.path.exists(os.path.join(dataset_folder, "normal_beat_Generate_lead_list_X.npy")):
        ECG_X_path = os.path.join(dataset_folder, "raw_data_X_train.npy")
        ECG_Y_path = os.path.join(dataset_folder, "raw_data_Y_train.npy")

        print("dataset: ", ECG_X_path)
        ECG_X_data = np.load(ECG_X_path)
        ECG_Y_data = np.load(ECG_Y_path)

        print("raw data shape: ", ECG_X_data.shape, ECG_Y_data.shape)

        # print(np.where(ECG_Y_data[:, 0] == 1))

        normal_ECG_X_data = ECG_X_data[np.where(ECG_Y_data[:, 0] == 1)]
        normal_ECG_Y_data = ECG_Y_data[np.where(ECG_Y_data[:, 0] == 1)]

        abnormal_ECG_X_data = ECG_X_data[np.where(ECG_Y_data[:, 0] == 0)]
        abnormal_ECG_Y_data = ECG_Y_data[np.where(ECG_Y_data[:, 0] == 0)]

        normal_beat_lead_list = [[], [], [], [], [], [], [], [], [], [], [], []]
        normal_y_data = [[], [], [], [], [], [], [], [], [], [], [], []]
        cnt = 0
        for ECG_leadList in normal_ECG_X_data:
            print(cnt, normal_ECG_Y_data[cnt])
            try:
                ECG_leadList = ECG_crop(ECG_leadList, avg_length=5000)
                beat_cnt = []
                beat_split_index, beat_padding_index, ECG_ll, ECG_rr = get_index_from_recording_12_lead(ECG_leadList,
                                                                                                        noise_leadID_list=None)

                for lead_index in range(12):
                    tmp_x = get_beat_from_recording_by_index(ECG_leadList[lead_index], beat_split_index)

                    beat_cnt.append(len(tmp_x))
                    for j in range(len(tmp_x)):
                        if tmp_x[j].size == 500:
                            normal_beat_lead_list[lead_index].append(tmp_x[j])
                            normal_y_data[lead_index].append(1)
                        else:
                            print(tmp_x[j].size)
            except:
                print("*** get beat error!", normal_ECG_Y_data[cnt])
            cnt += 1

        max_num = 0

        for i in range(12):
            print("lead:{}: \n".format(i + 1), len(normal_beat_lead_list), len(normal_beat_lead_list[i]),
                  len(normal_beat_lead_list[i][0]))
            if max_num < len(normal_beat_lead_list[i]):
                max_num = len(normal_beat_lead_list[i])
            print(len(normal_y_data[i]))

        print("normal beat max lead num: ", max_num)

        # 均衡各导联Beat数量
        for i in range(12):
            diff_num = max_num - len(normal_beat_lead_list[i])
            for j in range(diff_num):
                # 在当前导联所有beat下标中随机采样，直接复制，进行补充
                random_index = random.randint(0, len(normal_beat_lead_list[i]) - 1)
                normal_beat_lead_list[i].append(normal_beat_lead_list[i][random_index])
                normal_y_data[i].append(normal_y_data[i][random_index])

        abnormal_beat_lead_list = [[], [], [], [], [], [], [], [], [], [], [], []]
        abnormal_y_data = [[], [], [], [], [], [], [], [], [], [], [], []]
        abnormal_beat_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cnt = 0
        for ECG_leadList in abnormal_ECG_X_data:
            print(cnt, abnormal_ECG_Y_data[cnt])
            try:
                ECG_leadList = ECG_crop(ECG_leadList, avg_length=5000)

                beat_split_index, beat_padding_index, ECG_ll, ECG_rr = get_index_from_recording_12_lead(ECG_leadList,
                                                                                                        noise_leadID_list=None)

                for lead_index in range(12):
                    tmp_x = get_beat_from_recording_by_index(ECG_leadList[lead_index], beat_split_index)

                    for j in range(len(tmp_x)):
                        if tmp_x[j].size == 500:
                            abnormal_beat_lead_list[lead_index].append(tmp_x[j])
                            abnormal_y_data[lead_index].append(0)
                            abnormal_beat_cnt[lead_index] += 1
                            if abnormal_beat_cnt[lead_index] >= max_num * 0.1:
                                break
                        else:
                            print(tmp_x[j].size)
                if max(abnormal_beat_cnt) >= max_num * 0.1:
                    break
            except:
                print("*** get beat error!", abnormal_ECG_Y_data[cnt])
            cnt += 1

        # 均衡各abnormal导联Beat数量
        for i in range(12):
            diff_num = max(abnormal_beat_cnt) - len(abnormal_beat_lead_list[i])
            if diff_num > 0:
                for j in range(diff_num):
                    # 在当前导联所有beat下标中随机采样，直接复制，进行补充
                    random_index = random.randint(0, len(abnormal_beat_lead_list[i]) - 1)
                    abnormal_beat_lead_list[i].append(abnormal_beat_lead_list[i][random_index])
                    abnormal_y_data[i].append(abnormal_y_data[i][random_index])

        for i in range(12):
            print("Normal lead:{}: \n".format(i + 1), len(normal_beat_lead_list), len(normal_beat_lead_list[i]),
                  len(normal_beat_lead_list[i][0]))
            print("Abnormal lead:{}: \n".format(i + 1), len(abnormal_beat_lead_list), len(abnormal_beat_lead_list[i]),
                  len(abnormal_beat_lead_list[i][0]))

        np.save(os.path.join(dataset_folder, "normal_beat_Generate_lead_list_X.npy"), normal_beat_lead_list)
        np.save(os.path.join(dataset_folder, "normal_beat_Generate_lead_list_Y.npy"), normal_y_data)

        np.save(os.path.join(dataset_folder, "abnormal_beat_Generate_lead_list_X.npy"), abnormal_beat_lead_list)
        np.save(os.path.join(dataset_folder, "abnormal_beat_Generate_lead_list_Y.npy"), abnormal_y_data)

    else:
        print(str(os.path.join(dataset_folder, "normal_beat_Generate_lead_list_X.npy")) + " exists")

def read_CPSC_ECG_data_from_folder():
    # 构造整个CPSC数据的X和Y，并按照Stratified Sampling划分10折
    # 根据normal_csvPath读取noise lead < 2的Normal ECG和Abnormal ECG，构造X、Y
    # X：n * 12 * 5000
    # Y：n * 4 ([0/1, class id, file id, folder id])

    Y_list = [[], [], [], [], [], [], [], [], [], [], []]

    normal_csvPath = "./dataset/CPSC/normal_artificial.csv"
    normal_csvFile = open(normal_csvPath, "r")
    normal_reader = csv.reader(normal_csvFile)

    for item in normal_reader:
        lead_ID_list_str = (item[1].replace(' ', '')[1:-1]).split(',')
        lead_ID_list = []
        for item_lead in lead_ID_list_str:
            if len(item_lead) > 2:
                tmp_lead_ID = item_lead[1:-1]
                lead_ID_list.append(int(tmp_lead_ID))

        if len(lead_ID_list) <= 2:
            Y_list[1].append(item[0])

    print("CPSC Normal ECG num: ", len(Y_list[1]))

    class_cnt_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    class_cnt_list[1] = len(Y_list[1])

    label_csvPath = "./dataset/CPSC/reference.csv"
    label_csvFile = open(label_csvPath, "r")
    label_reader = csv.reader(label_csvFile)

    label_dict = {}
    for item in label_reader:
        label_dict[item[0]] = item[1]
        # normal_recordingList.append(item[0])
    # print("label num: ", len(label_dict))
    # print(label_dict)

    CPSC_data_TrainSet = "./dataset/CPSC/raw_ECGdata"
    files = os.listdir(CPSC_data_TrainSet)
    random.seed(2022)
    random.shuffle(files) # 打乱顺序

    cnt = 1
    normal_ECG_cnt = 0
    abnormal_ECG_cnt = 0
    for item in files:
        filename = item.split('.')[0]
        # print("recording cnt: {}, {}".format(cnt, int(filename[1:])))
        cnt += 1
        tmp_label = int(label_dict[filename])

        # print("raw label: ", label_dict[filename])
        if filename in Y_list[1]:
            normal_ECG_cnt += 1
        else:
            if tmp_label == 1:
                continue
            if abnormal_ECG_cnt == class_cnt_list[1]:
                continue
            # 保证各类别数据平衡
            if class_cnt_list[tmp_label] > class_cnt_list[1] / 8:
                continue
            else:
                class_cnt_list[tmp_label] += 1
            Y_list[tmp_label].append(filename)

            abnormal_ECG_cnt += 1

    print("class cnt: ", class_cnt_list)
    # print(Y_list[2])

    X = []
    Y = []
    cnt = -1
    for index in range(0, 10):
        print(index, Y_list[index])
        for item in Y_list[index]:
            cnt += 1
            fold_id = cnt % 10 + 1
            print(cnt, fold_id, item)
            ECG_leadList, tmp_age_sex_info = read_mat(os.path.join(CPSC_data_TrainSet, item + '.mat'), dataset="CPSC")
            ECG_leadList = ECG_crop(ECG_leadList, avg_length=5000)

            X.append(ECG_leadList)

            if index != 1:
                Y.append(np.array([0, index, int(item[1:]), fold_id]))
            else:
                Y.append(np.array([1, 1, int(item[1:]), fold_id]))

    X = np.array(X)
    Y = np.array(Y)
    for i in range(1, 11):
        print(i, len(np.where(Y[:,3] == i)[0]))
    print(Y)

    from sklearn.utils import shuffle
    X, Y = shuffle(X, Y, random_state=2022)
    print(Y)

    return X, Y


def read_AIWIN_ECG_data_from_folder():
    # 构造整个AIWIN数据的X和Y，并按照Stratified Sampling划分10折
    # 根据normal_csvPath读取noise lead < 2的Normal ECG和Abnormal ECG，构造X、Y
    # X：n * 12 * 5000
    # Y：n * 4 ([0/1, class id, file id, folder id])
    # (0正常心电图，1异常心电图）

    Y_list = [[], [], [], [], [], [], [], [], [], [], []]

    normal_csvPath = "./dataset/AIWIN/normal_artificial.csv"
    normal_csvFile = open(normal_csvPath, "r")
    normal_reader = csv.reader(normal_csvFile)

    for item in normal_reader:
        lead_ID_list_str = (item[1].replace(' ', '')[1:-1]).split(',')
        lead_ID_list = []
        for item_lead in lead_ID_list_str:
            if len(item_lead) > 2:
                tmp_lead_ID = item_lead[1:-1]
                lead_ID_list.append(int(tmp_lead_ID))

        if len(lead_ID_list) <= 2:
            Y_list[1].append(item[0])

    print("AIWIN Normal ECG num: ", len(Y_list[1]))

    class_cnt_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    class_cnt_list[1] = len(Y_list[1])

    label_csvPath = "./dataset/AIWIN/reference.csv"
    label_csvFile = open(label_csvPath, "r")
    label_reader = csv.reader(label_csvFile)

    label_dict = {}
    for item in label_reader:
        # (0正常心电图，1异常心电图）
        if item[1] == 'tag':
            continue
        label_dict[item[0]] = 1 - int(item[1])
        # normal_recordingList.append(item[0])
    # print("label num: ", len(label_dict))
    # print(label_dict)

    AIWIN_data_TrainSet = "./dataset/AIWIN/raw_ECGdata"
    files = os.listdir(AIWIN_data_TrainSet)
    random.seed(2022)
    random.shuffle(files) # 打乱顺序

    cnt = 1
    normal_ECG_cnt = 0
    abnormal_ECG_cnt = 0
    for item in files:
        filename = item.split('.')[0]
        # print("recording cnt: {}, {}".format(cnt, int(filename[1:])))
        cnt += 1
        tmp_label = int(label_dict[filename])

        # print("raw label: ", label_dict[filename])
        if filename in Y_list[1]:
            normal_ECG_cnt += 1
        else:
            if tmp_label == 1:
                continue
            if abnormal_ECG_cnt == class_cnt_list[1]:
                continue
            # 保证各类别数据平衡
            class_cnt_list[tmp_label] += 1
            Y_list[tmp_label].append(filename)

            abnormal_ECG_cnt += 1

    print("class cnt: ", class_cnt_list)
    # print(Y_list[2])

    X = []
    Y = []
    cnt = -1
    for index in range(0, 2):
        print(index, Y_list[index])
        for item in Y_list[index]:
            cnt += 1
            fold_id = cnt % 10 + 1
            print(cnt, fold_id, item)
            ECG_leadList, tmp_age_sex_info = read_mat(os.path.join(AIWIN_data_TrainSet, item + '.mat'), dataset="AIWIN")
            ECG_leadList = ECG_crop(ECG_leadList, avg_length=5000)

            X.append(ECG_leadList)
            Y.append(np.array([index, index, int(item[4:]) * 10000, fold_id]))

    X = np.array(X)
    Y = np.array(Y)
    for i in range(1, 11):
        # 打印每折的数量
        print(i, len(np.where(Y[:,3] == i)[0]))
    print(Y)

    from sklearn.utils import shuffle
    X, Y = shuffle(X, Y, random_state=2022)
    print(Y)

    return X, Y


def split_data_and_save(X, Y, dataset='CPSC', isShow=False):
    val_fold_list = [9, 10, 1, 2, 3, 4, 5, 6, 7, 8]
    test_fold_list = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for fold_index in range(10):
        val_fold = val_fold_list[fold_index]
        test_fold = test_fold_list[fold_index]

        data_folder = "./experiment/{}/dataset_val={}_test={}".format(dataset, val_fold, test_fold)
        print(data_folder)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)  # makedirs 创建文件时如果路径不存在会创建这个路径
            print("create folder: ", data_folder)

            # print(np.where(Y[:,3] == val_fold))

            # Train
            X_train = X[np.where((Y[:,3] != test_fold) & (Y[:,3]  != val_fold))]
            Y_train = Y[np.where((Y[:,3] != test_fold) & (Y[:,3]  != val_fold))]

            # Val
            X_val = X[np.where(Y[:,3] == val_fold)]
            Y_val = Y[np.where(Y[:,3] == val_fold)]

            # Test
            X_test = X[np.where(Y[:,3] == test_fold)]
            Y_test = Y[np.where(Y[:,3] == test_fold)]

            print("val fold: {}, test fold: {}".format(val_fold, test_fold))
            print("train size: ", X_train.shape, Y_train.shape)
            print("val size: ", X_val.shape, Y_val.shape)
            print("test size: ", X_test.shape, Y_test.shape)
            # print("Y test: \n", Y_test)

            np.save(os.path.join(data_folder, "raw_data_X_train.npy"), X_train)
            np.save(os.path.join(data_folder, "raw_data_Y_train.npy"), Y_train)

            np.save(os.path.join(data_folder, "raw_data_X_val.npy"), X_val)
            np.save(os.path.join(data_folder, "raw_data_Y_val.npy"), Y_val)

            np.save(os.path.join(data_folder, "raw_data_X_test.npy"), X_test)
            np.save(os.path.join(data_folder, "raw_data_Y_test.npy"), Y_test)
        else:
            print("folder is exist: ", data_folder)


def mix_CPSC_AIWIN_abnormal_from_CPSC_AIWIN():
    val_fold_list = [9, 10, 1, 2, 3, 4, 5, 6, 7, 8]
    test_fold_list = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for fold_index in range(10):
        val_fold = val_fold_list[fold_index]
        test_fold = test_fold_list[fold_index]

        data_folder = "./experiment/{}/dataset_val={}_test={}".format("CPSC_AIWIN_abnormal_from_CPSC_AIWIN", val_fold, test_fold)
        CPSC_data_folder = "./experiment/{}/dataset_val={}_test={}".format("CPSC", val_fold, test_fold)
        AIWIN_data_folder = "./experiment/{}/dataset_val={}_test={}".format("AIWIN", val_fold, test_fold)
        print(data_folder)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)  # makedirs 创建文件时如果路径不存在会创建这个路径
            print("create folder: ", data_folder)

            CPSC_X_train = np.load(os.path.join(CPSC_data_folder, "raw_data_X_train.npy"))
            CPSC_Y_train = np.load(os.path.join(CPSC_data_folder, "raw_data_Y_train.npy"))

            AIWIN_X_train = np.load(os.path.join(AIWIN_data_folder, "raw_data_X_train.npy"))
            AIWIN_Y_train = np.load(os.path.join(AIWIN_data_folder, "raw_data_Y_train.npy"))

            # Train
            X_train = np.vstack((CPSC_X_train, AIWIN_X_train))
            Y_train = np.vstack((CPSC_Y_train, AIWIN_Y_train))

            # Val
            CPSC_X_val = np.load(os.path.join(CPSC_data_folder, "raw_data_X_val.npy"))
            CPSC_Y_val = np.load(os.path.join(CPSC_data_folder, "raw_data_Y_val.npy"))

            AIWIN_X_val = np.load(os.path.join(AIWIN_data_folder, "raw_data_X_val.npy"))
            AIWIN_Y_val = np.load(os.path.join(AIWIN_data_folder, "raw_data_Y_val.npy"))

            # val
            X_val = np.vstack((CPSC_X_val, AIWIN_X_val))
            Y_val = np.vstack((CPSC_Y_val, AIWIN_Y_val))

            # Test
            CPSC_X_test = np.load(os.path.join(CPSC_data_folder, "raw_data_X_test.npy"))
            CPSC_Y_test = np.load(os.path.join(CPSC_data_folder, "raw_data_Y_test.npy"))

            AIWIN_X_test = np.load(os.path.join(AIWIN_data_folder, "raw_data_X_test.npy"))
            AIWIN_Y_test = np.load(os.path.join(AIWIN_data_folder, "raw_data_Y_test.npy"))

            # test
            X_test = np.vstack((CPSC_X_test, AIWIN_X_test))
            Y_test = np.vstack((CPSC_Y_test, AIWIN_Y_test))

            print("val fold: {}, test fold: {}".format(val_fold, test_fold))
            print("train size: ", X_train.shape, Y_train.shape)
            print("val size: ", X_val.shape, Y_val.shape)
            print("test size: ", X_test.shape, Y_test.shape)
            # print("Y test: \n", Y_test)

            # 保存数据集为numpy，方便对比实验读取
            np.save(os.path.join(data_folder, "raw_data_X_train.npy"), X_train)
            np.save(os.path.join(data_folder, "raw_data_Y_train.npy"), Y_train)

            np.save(os.path.join(data_folder, "raw_data_X_val.npy"), X_val)
            np.save(os.path.join(data_folder, "raw_data_Y_val.npy"), Y_val)

            np.save(os.path.join(data_folder, "raw_data_X_test.npy"), X_test)
            np.save(os.path.join(data_folder, "raw_data_Y_test.npy"), Y_test)
        else:
            print("folder is exist: ", data_folder)




if __name__ == '__main__':

    X, Y = read_CPSC_ECG_data_from_folder()
    split_data_and_save(X, Y, dataset='CPSC')

    X, Y = read_AIWIN_ECG_data_from_folder()
    split_data_and_save(X, Y, dataset='AIWIN')

    mix_CPSC_AIWIN_abnormal_from_CPSC_AIWIN()
