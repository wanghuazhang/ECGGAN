import os
import sys
import argparse
import numpy as np

import torch

# from ECG_options import ECG_Options
# from ecgClassifierModel import ecgClassifier
from model import MyGAN
from options import Options
from preProcess import get_Beat_Generate_dataset_Extract_beat_and_lead_label_from_npdata


def train_CGAN_model_item(opt, val_fold, test_fold):
    opt.dataset_folder = "./experiment/{}/dataset_val={}_test={}".format(opt.dataset, val_fold, test_fold)
    opt.log_save_path = os.path.join(opt.dataset_folder, "ECGGAN_log")

    opt.normal_data_TrainSetBeat_GAN_X = os.path.join(opt.dataset_folder, "normal_beat_Generate_lead_list_X.npy")
    opt.normal_data_TrainSetBeat_GAN_Y = os.path.join(opt.dataset_folder, "normal_beat_Generate_lead_list_Y.npy")
    opt.abnormal_data_TrainSetBeat_GAN_X = os.path.join(opt.dataset_folder, "abnormal_beat_Generate_lead_list_X.npy")
    opt.abnormal_data_TrainSetBeat_GAN_Y = os.path.join(opt.dataset_folder, "abnormal_beat_Generate_lead_list_Y.npy")

    if not os.path.exists(opt.normal_data_TrainSetBeat_GAN_X):
        get_Beat_Generate_dataset_Extract_beat_and_lead_label_from_npdata(dataset_folder=opt.dataset_folder)

    normal_mode_list = ["no_normal", "one_lead", "all_lead"]

    if opt.dataset == "PTB-XL":
        opt.batch_size = 256

    mymodel = MyGAN(opt, mode="train")
    mymodel.train_beat_12_lead(isSave=True)

    if not mymodel.have_trained:
        auc, prec, rec, best_f1, best_thr = mymodel.validate()
        print("beat test:\nauc{:.4f} precision {:.4f} recall {:.4f} best_f1 {:.4f} best_threshold {:.4f}".format(auc,
                                                                                                                 prec,
                                                                                                                 rec,
                                                                                                                 best_f1,
                                                                                                                 best_thr))
    else:
        print(mymodel.log_folder + " is exists.")

def load_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, default="train_CGAN") # train_CGAN, train_anomaly_detection_module, test_ECG
    parser.add_argument("--dataset", type=str, required=True, default="CPSC") # CPSC, AIWIN, Mixed-set
    parser.add_argument("--val_fold", type=int, required=True, default=9) # val=[9, 10, 1, 2, 3, 4, 5, 6, 7, 8], test = (val % 10) + 1
    parser.add_argument("--test_fold", type=int, required=True, default=10)
    # parser.add_argument("--dataset", type=str, required=True, default="CPSC")
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = load_argparse()
    print(args)

    opt = Options()
    opt.dataset = args['dataset']

    if args['mode'] == "train_CGAN":
        # 指定val & test
        train_CGAN_model_item(opt, args['val_fold'], args['test_fold'])

    if args['mode'] == "train_anomaly_detection_module":
        pass
