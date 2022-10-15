import os
import scipy.io as scio
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import torch

def class_id_to_class_name(class_id=1):
    class_name = {}
    class_name[1] = "Normal"
    class_name[2] = "Atrial fibrillation (AF)"
    class_name[3] = "First-degree atrioventricular block (I-AVB)"
    class_name[4] = "Left bundle branch block (LBBB)"
    class_name[5] = "Right bundle branch block (RBBB)"
    class_name[6] = "Premature atrial contraction (PAC)"
    class_name[7] = "Premature ventricular contraction (PVC)"
    class_name[8] = "ST-segment depression (STD)"
    class_name[9] = "ST-segment elevated (STE)"
    return class_name[class_id]

def read_mat(file_path, dataset="CPSC"):
    if dataset == "CPSC":
        data_raw = scio.loadmat(file_path)
        data_mat = data_raw['ECG'][0][0]
        # 读取性别
        sex = 0
        if data_mat[0] == 'Male':
            sex = 1
        # 读取年龄
        age = data_mat[1][0][0]
        ECG_leadList = data_mat[2]
        time = ECG_leadList[0].size / 500
        tmp_other_info = {'sex': sex, 'age': age}
        # tmp_other_info['time'] = time
        # print(type(ECG_leadList), ECG_leadList.shape)
        return ECG_leadList, tmp_other_info
    elif dataset == "AIWIN":
        ECG_leadList = scio.loadmat(file_path)['ecgdata']
        return ECG_leadList, None
    elif dataset == "PTB-XL":
        data = wfdb.rdsamp(file_path)
        ECG_recording = data[0].transpose((1, 0))
        return ECG_recording


def beats_show(beats, row = 3, col = 3, title="beats show:"):
    plt.figure()
    # row = int((len(beats) + 2) / 3)  # 行

    # print(row)
    for i in range(row):
        for j in range(col):

            index = i * col + j
            if index >= len(beats):
                break
            plt.subplot(row, col, index + 1)
            plt.plot(beats[index])
            # plt.ylim(-1, 1)
            # plt.axis([0, 500, -1, 1])

    plt.suptitle(title)

    plt.show()


def show_heat_from_beat(ori_beat, gen_beat, title=None, isShow=False, isSave=False, save_path=None):
    difference = torch.pow(ori_beat - gen_beat, 2)
    heat_normalize = (difference - torch.min(difference)) / (torch.max(difference) - torch.min(difference))
    # print(heat_normalize.shape)

    x_points = np.arange(ori_beat.shape[1])
    if ori_beat.shape[1] <= 500:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 4), gridspec_kw={'height_ratios': [5, 1], })
    else:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 2), gridspec_kw={'height_ratios': [5, 1], })
    plt.ylim(-1, 1)
    ax[0].plot(x_points, ori_beat.view(-1), 'k-')
    ax[0].plot(x_points, gen_beat.view(-1), 'r-')

    ax[1].imshow(heat_normalize, cmap="jet", aspect="auto", vmin=0, vmax=0.1)
    ax[1].set_yticks([])
    if title:
        plt.title(title)

    if isSave:
        plt.savefig(save_path)

    if isShow:
        plt.show()
    else:
        plt.close()
