import os
import sys
import argparse
import numpy as np
from multiprocessing import Pool

import torch

from ECG_options import ECG_Options
from ecgClassifierModel import ecgClassifier
from model import MyGAN
from options import Options
from preProcess import get_Beat_Generate_dataset_Extract_beat_and_lead_label_from_npdata

def train_CGAN_model_item(opt, val_fold, test_fold):
    opt.dataset_folder = "/home2/wanghuazhang/ECGGAN/experiment/{}/dataset_val={}_test={}".format(
        opt.dataset, val_fold, test_fold)
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
    if opt.model_mode == "CGAN" or opt.model_mode == "CGanomaly":
        mymodel.train_beat_12_lead(isSave=True)
    else:
        mymodel.train_beat_lead(isSave=True)
    if not mymodel.have_trained:
        auc, prec, rec, best_f1, best_thr = mymodel.validate()
        print("beat test:\nauc{:.4f} precision {:.4f} recall {:.4f} best_f1 {:.4f} best_threshold {:.4f}".format(auc,
                                                                                                                 prec,
                                                                                                                 rec,
                                                                                                                 best_f1,
                                                                                                                 best_thr))
    else:
        print(mymodel.log_folder + " is exists.")




if __name__ == '__main__':

    opt = Options()



