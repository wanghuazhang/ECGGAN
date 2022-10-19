import os
import sys
import argparse
import numpy as np

from ECG_options import ECG_Options
from ecgClassifierModel import ecgClassifier
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


def train_test_detect_module_item(ECG_opt, opt, val_fold, test_fold):
    print("test_GAN_item", val_fold, test_fold)
    ECG_opt.val_fold = val_fold
    ECG_opt.test_fold = test_fold
    model_path_G = "./experiment/{}/dataset_val={}_test={}/ECGGAN_log/Experiment_{}_label={}_gen={}_feature={}_{}_batchsize=128_drop={}_normal_False/epoch_100_Generator_model.pkl".format(
        ECG_opt.dataset, val_fold, test_fold, ECG_opt.dataset, opt.G_label_loss_weight,
        opt.G_gen_loss_weight, opt.G_feature_loss_weight, ECG_opt.experiment_date, opt.dropout_ratio)
    model_path_D = "./experiment/{}/dataset_val={}_test={}/ECGGAN_log/Experiment_{}_label={}_gen={}_feature={}_{}_batchsize=128_drop={}_normal_False/epoch_100_Discriminator_model.pkl".format(
        ECG_opt.dataset, val_fold, test_fold, ECG_opt.dataset, opt.G_label_loss_weight,
        opt.G_gen_loss_weight, opt.G_feature_loss_weight, ECG_opt.experiment_date, opt.dropout_ratio)

    print(model_path_G)

    ECG_gen_model = MyGAN(opt, mode='ECG gen', model_path_G=model_path_G, model_path_D=model_path_D)

    ECG_classifier_model = ecgClassifier(ECG_opt, mode='ECG matrix train', ECG_gen_model=ECG_gen_model,
                                         Generator_model_path=model_path_G)

    F1_idx, F1_list_val, F1_list_test, F1_list_train, auc_list_val, auc_list_test, auc_list_train = ECG_classifier_model.train_ECG_Matrix_Classifier_Model(isSave=False)

    return F1_idx, F1_list_val, F1_list_test, F1_list_train, auc_list_val, auc_list_test, auc_list_train


def load_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, default="train_CGAN") # reconstruction_model, anomaly_detection_module, ECGGAN
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

    ECG_opt = ECG_Options()

    if args['mode'] == "reconstruction_model" or args['mode'] == "ECGGAN":
        # 指定val & test
        train_CGAN_model_item(opt, args['val_fold'], args['test_fold'])

    if args['mode'] == "anomaly_detection_module" or args['mode'] == "ECGGAN":
        # 指定val & test
        train_test_detect_module_item(ECG_opt, opt, args['val_fold'], args['test_fold'])
