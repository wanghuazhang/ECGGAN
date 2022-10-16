import csv
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import dataloader, TensorDataset, DataLoader
import tensorwatch as tw
from torchviz import make_dot
import hiddenlayer as h
from net import weights_init, Generator_CGAN, Discriminator_CGAN
from preProcess import get_beat_from_recording_by_index, get_index_from_recording_12_lead
from utils import read_mat, beats_show, show_heat_from_beat, ECG_crop


class MyGAN():
    def __init__(self, opt, mode=None, lead=None, model_path_G=None, model_path_D=None, data_folder=None):
        self.opt = opt
        self.mode = mode
        self.traindata = self.opt.dataset

        self.data_folder = data_folder

        self.have_trained = False

        # 设置保存的model类似：iter代表每次auc更高的模型都被单独保存，only代表只有最好的一次模型被保存
        self.save_train_model = "iter"
        self.f_train_log = None
        print("ECG Gen Model mode:{}, dataset:{}".format(mode, self.traindata))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化CGAN训练时lead_label的one hot编码
        self.lead_label_1hots = torch.zeros(12, 12, self.opt.beat_size)
        for i in range(12):
            self.lead_label_1hots[i][i] = torch.ones(self.opt.beat_size)
        self.lead_label_1hots = self.lead_label_1hots.to(self.device)

        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()

        if self.mode == "train":
            self.best_auc = 0
            self.prec = 0
            self.rec = 0
            self.best_f1 = 0
            self.best_thr = 0
            self.best_epoch = 0
            self.best_loss = 1000
            self.sava_mode_auc = False
            self.sava_mode_loss = False

            self.last_val_loss = 0

            self.save_g_label_loss = []
            self.save_g_gen_loss = []
            self.save_g_feature_loss = []

            self.data_loader = self.load_train_beats_data(lead)

            self.G = Generator_CGAN(self.opt)

            self.D = Discriminator_CGAN(self.opt)

            if (self.device.type == 'cuda') and (self.opt.ngpu > 1):
                self.G = nn.DataParallel(self.G, list(range(self.opt.ngpu)))
            if (self.device.type == 'cuda') and (self.opt.ngpu > 1):
                self.D = nn.DataParallel(self.D, list(range(self.opt.ngpu)))

            self.G.to(self.device)
            self.D.to(self.device)
            self.G.apply(weights_init)
            self.D.apply(weights_init)

            self.real_labels = torch.ones(self.opt.batch_size, 1, device=self.device)
            self.fake_labels = torch.zeros(self.opt.batch_size, 1, device=self.device)

            self.G_optim = optim.RMSprop(self.G.parameters(), lr=self.opt.lr)
            self.D_optim = optim.RMSprop(self.D.parameters(), lr=self.opt.lr)

            self.log_folder = ""

            self.fixed_x = None
            self.fixed_y = None

            # train_time是每一个epoch训练时间
            self.train_time = []
            # self.G = self.load_CGAN_model()
        elif mode == "ECG test":
            self.G = self.load_CGAN_model(model_path_G)

            self.ECG_test_dataloader = self.load_ECG_test_data()

        elif mode == "ECG gen":
            # ECG 生成器加载
            self.G = self.load_CGAN_model(model_path_G)
            self.D = self.load_CGAN_model(model_path_D)
            self.G.to(self.device)
            self.D.to(self.device)
            if data_folder is not None:
                self.load_ECG_test_data()

    def normalize(self, seq):
        if self.opt.isNormalization:
            if self.opt.preprocess == "max_min":
                if np.max(seq) == np.min(seq):
                    return seq
                return 2 * (seq - np.min(seq)) / (np.max(seq) - np.min(seq)) - 1
            else:
                return seq
        else:
            return seq

    def load_train_beats_data(self, lead):
        print("Load Train Beats Data: ")
        normal_DataSet_X = None
        normal_DataSet_Y = None

        abnormal_DataSet_X = None
        abnormal_DataSet_Y = None

        np_normal_DataSet_X = np.load(self.opt.normal_data_TrainSetBeat_GAN_X, allow_pickle=True)
        np_normal_DataSet_Y = np.load(self.opt.normal_data_TrainSetBeat_GAN_Y, allow_pickle=True)
        np_abnormal_DataSet_X = np.load(self.opt.abnormal_data_TrainSetBeat_GAN_X, allow_pickle=True)
        np_abnormal_DataSet_Y = np.load(self.opt.abnormal_data_TrainSetBeat_GAN_Y, allow_pickle=True)

        print("np_normal_DataSet_X:", np_normal_DataSet_X.shape)
        print("np_normal_DataSet_Y:", np_normal_DataSet_Y.shape)
        print("np_abnormal_DataSet_X:", np_abnormal_DataSet_X.shape)
        print("np_abnormal_DataSet_Y:", np_abnormal_DataSet_Y.shape)

        # 构造12导联beat数据，同时构造对应beat所在导联的lead label
        for i in range(12):
            print("loda lead {}: {}".format(i + 1, np_normal_DataSet_X[i].shape))
            tmp_normal_X = np.array(np_normal_DataSet_X[i]).reshape((-1, self.opt.nc, self.opt.beat_size))

            tmp_normal_Y = np.array(np_normal_DataSet_Y[i])

            # 构造导联的label, 1-12，注意：这里的lead对应0-11
            # For normal data:
            tmp_lead_label_Y = np.zeros(tmp_normal_Y.shape[0])
            for j in range(tmp_lead_label_Y.shape[0]):
                tmp_lead_label_Y[j] = i

            tmp_normal_Y = np.hstack((tmp_normal_Y.reshape(tmp_normal_Y.shape[0], 1),
                                      tmp_lead_label_Y.reshape(tmp_normal_Y.shape[0], 1)))

            # For abnormal data:
            tmp_abnormal_X = np.array(np_abnormal_DataSet_X[i]).reshape((-1, self.opt.nc, self.opt.beat_size))

            tmp_abnormal_Y = np.array(np_abnormal_DataSet_Y[i])

            tmp_lead_label_Y = np.zeros(tmp_abnormal_Y.shape[0])
            for j in range(tmp_lead_label_Y.shape[0]):
                tmp_lead_label_Y[j] = i

            tmp_abnormal_Y = np.hstack((tmp_abnormal_Y.reshape(tmp_abnormal_Y.shape[0], 1),
                                        tmp_lead_label_Y.reshape(tmp_abnormal_Y.shape[0], 1)))

            if normal_DataSet_X is None:
                normal_DataSet_X = tmp_normal_X
                normal_DataSet_Y = tmp_normal_Y

                abnormal_DataSet_X = tmp_abnormal_X
                abnormal_DataSet_Y = tmp_abnormal_Y
            else:
                normal_DataSet_X = np.vstack((normal_DataSet_X, tmp_normal_X))
                normal_DataSet_Y = np.vstack((normal_DataSet_Y, tmp_normal_Y))

                abnormal_DataSet_X = np.vstack((abnormal_DataSet_X, tmp_abnormal_X))
                abnormal_DataSet_Y = np.vstack((abnormal_DataSet_Y, tmp_abnormal_Y))

        print("Dataset size: ")
        print("Normal beat size: ", normal_DataSet_X.shape)
        print("Abnormal beat size: ", abnormal_DataSet_X.shape)

        if self.opt.isNormalization:
            print("normalize")
            for i in range(normal_DataSet_X.shape[0]):
                for j in range(self.opt.nc):
                    normal_DataSet_X[i][j] = self.normalize(normal_DataSet_X[i][j])
            for i in range(abnormal_DataSet_X.shape[0]):
                for j in range(self.opt.nc):
                    abnormal_DataSet_X[i][j] = self.normalize(abnormal_DataSet_X[i][j])

        train_x, test_x, train_y, test_y = train_test_split(normal_DataSet_X, normal_DataSet_Y, test_size=0.1,
                                                            random_state=self.opt.random_state)

        # 将normal、abnormal混合，从而构造验证集和测试集
        # 保证normal ： abnormal = 1： 1
        test_x = np.concatenate([test_x, abnormal_DataSet_X[0:test_x.shape[0]]])
        test_y = np.concatenate([test_y, abnormal_DataSet_Y[0:test_y.shape[0]]])

        # Beat reconstruction model doesn't need val dataset.
        val_x = test_x
        val_y = test_y

        print("train size: ", train_x.shape)
        print("val size: ", val_x.shape)
        print("test size: ", test_x.shape)

        train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
        test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        train_loader = dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            drop_last=True
        )

        val_loader = dataloader.DataLoader(
            dataset=val_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            drop_last=True
        )

        test_loader = dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            drop_last=True
        )

        print("Load End.")

        data_loader = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }

        return data_loader

    def load_ECG_test_data(self):

        train_X_path = os.path.join(self.data_folder, "raw_data_X_train.npy")
        train_Y_path = os.path.join(self.data_folder, "raw_data_Y_train.npy")
        val_X_path = os.path.join(self.data_folder, "raw_data_X_val.npy")
        val_Y_path = os.path.join(self.data_folder, "raw_data_Y_val.npy")
        test_X_path = os.path.join(self.data_folder, "raw_data_X_test.npy")
        test_Y_path = os.path.join(self.data_folder, "raw_data_Y_test.npy")
        print(train_X_path)

        raw_data_X_train = np.load(train_X_path)
        raw_data_Y_train = np.load(train_Y_path)
        raw_data_X_val = np.load(val_X_path)
        raw_data_Y_val = np.load(val_Y_path)
        raw_data_X_test = np.load(test_X_path)
        raw_data_Y_test = np.load(test_Y_path)

        print("Train size: ", raw_data_X_train.shape)
        print("Val size: ", raw_data_X_val.shape)
        print("Test size: ", raw_data_X_test.shape)

        self.raw_data_X_train = raw_data_X_train
        self.raw_data_Y_train = raw_data_Y_train
        self.raw_data_X_val = raw_data_X_val
        self.raw_data_Y_val = raw_data_Y_val
        self.raw_data_X_test = raw_data_X_test
        self.raw_data_Y_test = raw_data_Y_test

        train_dataset = TensorDataset(torch.from_numpy(raw_data_X_train), torch.from_numpy(raw_data_Y_train))
        val_dataset = TensorDataset(torch.from_numpy(raw_data_X_val), torch.from_numpy(raw_data_Y_val))
        test_dataset = TensorDataset(torch.from_numpy(raw_data_X_test), torch.from_numpy(raw_data_Y_test))

        # 训练数据装载
        train_loader = dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            drop_last=False
        )

        val_loader = dataloader.DataLoader(
            dataset=val_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            drop_last=False
        )

        test_loader = dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,
            drop_last=False
        )

        ECG_dataloader = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }

        print("Load ECG Data End.")

        return ECG_dataloader

    def train_beat_12_lead(self, isSave=True, lead=0):

        if isSave:
            self.create_log_folder()

        if self.have_trained:
            print(self.log_folder + " is exists.")
            return

        print("Starting Training Loop...")
        start_time = time.time()
        gen_all_beats = []
        for epoch in range(self.opt.num_epochs):
            G_loss_run = 0.0
            D_loss_run = 0.0

            tmp_G_label_loss = 0.0
            tmp_G_gen_loss = 0.0
            tmp_G_feature_loss = 0.0
            tmp_G_encoder2_loss = 0.0

            epoch_start_time = time.time()
            self.G.train()
            self.D.train()
            torch.set_grad_enabled(True)
            for i, data in enumerate(self.data_loader["train"]):
                X = torch.as_tensor(data[0], dtype=torch.float32)  # [banch_size, 1, 500]
                Y = torch.as_tensor(data[1], dtype=torch.long).t()

                if i % random.randint(1, 10) == 0:
                    self.fixed_x = X
                    self.fixed_y = Y

                X = X.to(self.device)
                Y = Y.to(self.device)

                lead_label = self.lead_label_1hots[Y[1]]

                for parm in self.D.parameters():
                    parm.data.clamp_(-self.opt.clamp_num, self.opt.clamp_num)

                # ----- train D -----
                self.D_optim.zero_grad()

                # Train real
                D_real, D_real_feature = self.D(X, lead_label)
                D_real_loss = self.bce_criterion(D_real, self.real_labels)

                # Train fake
                fake_imgs, G_fake_feature = self.G(X, lead_label)

                D_fake, D_fake_feature = self.D(fake_imgs, lead_label)
                D_fake_loss = self.bce_criterion(D_fake, self.fake_labels)

                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                self.D_optim.step()

                # ------ train G -------
                if (i + 1) % self.opt.n_critic == 0:
                    self.G_optim.zero_grad()

                    gen_imgs, G_fake_feature = self.G(X, lead_label)
                    D_fake, D_fake_feature = self.D(gen_imgs, lead_label)
                    # label loss
                    G_label_loss = self.bce_criterion(D_fake, self.real_labels)

                    # gen loss
                    G_gen_loss = self.mse_criterion(gen_imgs, X)

                    # feature loss
                    D_real, D_real_feature = self.D(X, lead_label)
                    G_feature_loss = self.mse_criterion(G_fake_feature, D_real_feature)
                    # G_feature_loss = self.mse_criterion(D_fake_feature, D_real_feature)

                    # WGAN Adversarial loss
                    # G_loss = -torch.mean(self.D(gen_imgs))

                    tmp_G_label_loss += G_label_loss.item() * self.opt.G_label_loss_weight
                    tmp_G_gen_loss += G_gen_loss.item() * self.opt.G_gen_loss_weight
                    tmp_G_feature_loss += G_feature_loss.item() * self.opt.G_feature_loss_weight

                    G_loss = G_label_loss * self.opt.G_label_loss_weight + G_gen_loss * self.opt.G_gen_loss_weight \
                             + G_feature_loss * self.opt.G_feature_loss_weight

                    G_loss.backward()
                    self.G_optim.step()
                    G_loss_run += G_loss.item()

                D_loss_run += D_loss.item()

            self.train_time.append(time.time() - epoch_start_time)

            time_log_content = "Epoch {}: training time: {:.2f}, Total time: {:.2f}, avg time: {:.2f}".format(epoch,
                                                                                                              self.train_time[
                                                                                                                  epoch],
                                                                                                              sum(
                                                                                                                  self.train_time),
                                                                                                              sum(
                                                                                                                  self.train_time) / (
                                                                                                                      epoch + 1))
            self.f_train_log.write(time_log_content)
            self.f_train_log.write('\n')

            loss_log_content = 'Epoch {}: G_loss: {}, D_loss: {}\n'.format(epoch, G_loss_run / (i + 1),
                                                                           D_loss_run / (i + 1))
            self.f_train_log.write(loss_log_content)

            tmp_loss_log_content = "Epoch {}: G_label_lossL {}, G_gen_loss: {}, G_feature_loss: {}, G_encoder2_loos: {}\n"\
                .format(epoch, tmp_G_label_loss / (i + 1), tmp_G_gen_loss / (i + 1), tmp_G_feature_loss / (i + 1), tmp_G_encoder2_loss / (i + 1))
            self.f_train_log.write(tmp_loss_log_content)

            self.save_g_label_loss.append(tmp_G_label_loss / (i + 1))
            self.save_g_gen_loss.append(tmp_G_gen_loss / (i + 1))
            self.save_g_feature_loss.append(tmp_G_feature_loss / (i + 1))

            print(time_log_content)
            print(loss_log_content)
            print(tmp_loss_log_content)

            # get auc、th、f1 for each epoch
            auc, prec, rec, best_f1, best_thr = self.validate()
            if self.best_auc < auc:
                self.best_auc = auc
                self.prec = prec
                self.rec = rec
                self.best_f1 = best_f1
                self.best_thr = best_thr
                self.best_epoch = epoch
                if isSave and self.sava_mode_auc:
                    if self.save_train_model == "iter":
                        model_title = "auc_{:.4f}_".format(self.best_auc)
                        self.save_model(model_title)
                    else:
                        self.save_model()

            if epoch in [50, 80, 100, 118]:
                model_title = "epoch_{}_".format(epoch)
                self.save_model(model_title)

            auc_log_content = "Epoch {}: auc{:.4f} precision {:.4f} recall {:.4f} best_f1 {:.4f} best_threshold {:.4f}, best_auc={:.4f} in epoch{}".format(
                epoch, auc, prec, rec, best_f1, best_thr, self.best_auc, self.best_epoch)
            self.f_train_log.write(auc_log_content)
            self.f_train_log.write('\n')

            print(auc_log_content)

        print("End Training.")
        self.f_train_log.close()

        # 打印loss曲线
        x_axix = list(range(len(self.save_g_label_loss)))
        # plt.ylim(0.5, 1)
        plt.title('loss Result')
        plt.plot(x_axix, self.save_g_label_loss, color='green', label='g label loss')
        plt.plot(x_axix, self.save_g_gen_loss, color='red', label='g gen loss')
        plt.plot(x_axix, self.save_g_feature_loss, color='blue', label='g feature loss')
        plt.legend()  # 显示图例

        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(self.log_folder, "train_loss.jpg"))
        plt.show()

    def create_log_folder(self):
        log_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
        print("Time:", log_time)
        # log_file_name = "Experiment_" + self.traindata + "_" + log_time
        log_file_name = "Experiment_{}".format(self.traindata) \
                        + "_label={}_gen={}_feature={}".format(self.opt.G_label_loss_weight, self.opt.G_gen_loss_weight, self.opt.G_feature_loss_weight)

        log_file_name += "_{}_batchsize={}_drop={}_normal_{}".format(self.opt.experiment_date, self.opt.batch_size, self.opt.dropout_ratio, self.opt.isNormalization)

        self.log_folder = os.path.join(self.opt.log_save_path, log_file_name)

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
            print("create folde: ", self.log_folder)
        if os.path.exists(os.path.join(self.log_folder, "epoch_100_Generator_model.pkl")):
            print(self.log_folder + " is exists.")
            self.have_trained = True
            return

        self.img_save_folder = os.path.join(self.log_folder, "train_Img")
        if not os.path.exists(self.img_save_folder):
            os.makedirs(self.img_save_folder)
            print("create folde: ", self.img_save_folder)

        parameter_setting = "\n".join([str([attr, value]) for attr, value in self.opt.__dict__.items()])

        print(parameter_setting)
        self.f_train_log = open(os.path.join(self.log_folder, "train_log.txt"), 'a')
        self.f_train_log.write(parameter_setting)
        self.f_train_log.write('\n')

        data_info = "train size: {}\nval size: {}\ntest size: {}\n".format(
            str(self.data_loader['train'].dataset.tensors[0].shape),
            str(self.data_loader['val'].dataset.tensors[0].shape),
            str(self.data_loader['test'].dataset.tensors[0].shape))
        print(data_info)
        self.f_train_log.write(data_info)

        print("Model Structure: ", self.G)
        self.f_train_log.write("Generator_CGAN Model Structure: \n")
        self.f_train_log.write(str(self.G))
        self.f_train_log.write('\n')
        para_content = self.get_parameter_number(self.G)

        print("Model Structure: ", self.D)
        self.f_train_log.write("Discriminator_CGAN Model Structure: \n")
        self.f_train_log.write(str(self.D))
        self.f_train_log.write('\n')
        para_content = self.get_parameter_number(self.D)

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        para_content = 'Total: {}, Trainable: {}'.format(total_num, trainable_num)
        print(para_content)
        self.f_train_log.write(para_content)
        self.f_train_log.write('\n')
        return para_content

    def save_model(self, title=None):
        D_file_path = os.path.join(self.log_folder, title + "Discriminator_model.pkl")
        G_file_path = os.path.join(self.log_folder, title + "Generator_model.pkl")
        # 保存整个网络
        torch.save(self.D, D_file_path)
        torch.save(self.G, G_file_path)
        self.opt.D_model_path = D_file_path
        self.opt.G_model_path = G_file_path

    def validate(self, dataset='val', mode='beat'):
        print("validate dataset:", dataset)
        if dataset == 'val':
            data_loader = self.data_loader["val"]
        elif dataset == 'test':
            data_loader = self.data_loader["test"]
        else:
            data_loader = self.data_loader["train"]

        y, y_pred = self.predict(data_loader)

        auc = metrics.roc_auc_score(y, y_pred)

        precs, recs, thrs = metrics.precision_recall_curve(y, y_pred)

        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]

        prec = precs[np.argmax(f1s)]
        rec = recs[np.argmax(f1s)]
        best_f1 = np.max(f1s)
        best_thr = thrs[np.argmax(f1s)]
        # print(auc, prec, rec, best_f1, best_thr)
        return auc, prec, rec, best_f1, best_thr

    def predict(self, dataloader_, scale=True):
        print("predicting")
        self.G.eval()
        self.D.eval()
        torch.set_grad_enabled(False)
        with torch.no_grad():
            ones = torch.ones(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            beat_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            y_labels = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long, device=self.device)
            fake_features = torch.zeros(size=(len(dataloader_.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            G_loss_run = 0.0

            for i, data in enumerate(dataloader_, 0):
                X = torch.as_tensor(data[0], dtype=torch.float32)
                X = X.to(self.device)

                Y = torch.as_tensor(data[1], dtype=torch.long).t()
                Y = Y.to(self.device)
                lead_label = self.lead_label_1hots[Y[1]]

                fake_imgs, G_fake_feature = self.G(X, lead_label)

                if self.opt.isSave_val_loss:
                    D_fake, D_fake_feature = self.D(fake_imgs, lead_label)

                    # label loss
                    G_label_loss = self.bce_criterion(D_fake, self.real_labels)

                    # gen loss
                    G_gen_loss = self.mse_criterion(fake_imgs, X)

                    # feature loss
                    D_real, D_real_feature = self.D(X, lead_label)

                    G_feature_loss = self.mse_criterion(G_fake_feature, D_real_feature)

                    G_loss = G_label_loss.item() * self.opt.G_label_loss_weight + \
                             G_gen_loss.item() * self.opt.G_gen_loss_weight + \
                             G_feature_loss.item() * self.opt.G_feature_loss_weight
                    G_loss_run += G_loss

                error = torch.mean(torch.pow((X.view(self.opt.batch_size, -1) - fake_imgs.view(self.opt.batch_size, -1)), 2), dim=1)

                beat_scores[i * self.opt.batch_size: i * self.opt.batch_size + error.size(0)] = error.reshape(error.size(0))

                y_labels[i * self.opt.batch_size: i * self.opt.batch_size + error.size(0)] = Y[0].reshape(error.size(0))
                fake_features[i * self.opt.batch_size: i * self.opt.batch_size + error.size(0), : ] = G_fake_feature.reshape(error.size(0), self.opt.nz)

            if scale:
                beat_scores = (beat_scores - torch.min(beat_scores)) / (torch.max(beat_scores) - torch.min(beat_scores))

            beat_scores = ones - beat_scores

            y = y_labels.cpu().numpy()
            y_pred = beat_scores.cpu().numpy()
            return y, y_pred

    def load_CGAN_model(self, model_path=None):
        if model_path is None:
            model_path = self.opt.G_model_path

        print("load CGAN model:", model_path)
        return torch.load(model_path).to(self.device)

    def gen_ECG_item(self, data=None, index=None, isShow=False, de_baseline=False, isLoss=False, beat_gen_normalize_mode="no_normal"):
        # de_baseline：代表是否减去基线数值，False使用原始数值，True减去基线数值
        if data:
            ECG_leadList = data[0]
            label = data[1]
        elif index is not None:
            X = self.raw_data_X_test
            Y = self.raw_data_Y_test
            ECG_leadList = X[index]
            label = Y[index][0]

            if isShow:
                title = "label: {}, class: {}, file: {}".format(Y[index][0], Y[index][1], Y[index][2])

                plt.figure(figsize=(10, 15))

                for i in range(12):
                    # plt.ylim(-1, 1)
                    plt.subplot(12, 1, i + 1)
                    plt.plot(ECG_leadList[i])
                    # plt.title("lead:" + str(i + 1))

                plt.suptitle(title)
                plt.show()

        data_x, data_y, beat_split_index, beat_padding_index, ECG_ll, ECG_rr = self.ECG_split_to_beats(
            ECG_leadList=ECG_leadList, label=label, isShow=True, beat_gen_normalize_mode=beat_gen_normalize_mode)
        exit()
        # if beat_gen_normalize_mode == "Beat_normal":
        #     for i in range(12): # 12 lead
        #         print(data_x[i].shape)
        #         for j in range(data_x[i].shape[0]):
        #             for k in range(self.opt.nc):
        #                 print(data_x[i][j][k].shape)
        #                 data_x[i][j][k] = self.normalize(data_x[i][j][k])

        if isLoss:
            gen_data_x, g_label_loss, g_feature_loss, g_gen_loss = self.gen_beat(data_x, data_y, isShow=False,
                                                                                 isLoss=isLoss)
            tmp_content = "simple g_label_loss: {}, g_gen_loss: {}, g_feature_loss: {}".format(g_label_loss, g_gen_loss,
                                                                                               g_feature_loss)
            print(tmp_content)
            if self.f_train_log is not None:
                self.f_train_log.write(tmp_content)
                self.f_train_log.write('\n')
        else:
            gen_data_x = self.gen_beat(data_x, data_y, isShow=False, isLoss=isLoss)

        # print("gen data x size: ", gen_data_x[0].shape)

        if beat_gen_normalize_mode == "Beat_normal":
            # 对切割处理的每个Beat归一化操作
            # ori_data_x = gen_data_x
            for i in range(12):
                data_x[i] = torch.squeeze(data_x[i])
                # 减去了baseline，会让归一化之后的ECG有所偏移
                # print(data_x[i].shape, np.max(data_x[i].numpy()), np.min(data_x[i].numpy()))
            ori_data_x = self.beat_de_crop(data_x, beat_padding_index)
            ori_baseline_value_list = self.get_baseline_value(ori_data_x)
            print("ori_baseline_value_list: ", ori_baseline_value_list)

            ori_ECG_list = self.beats_connect_to_ECG(ori_data_x)
        elif beat_gen_normalize_mode == "no_normal":
            ori_baseline_value_list = self.get_baseline_value(ECG_leadList, beat_split_index)
            # print("ori_baseline_value_list: ", ori_baseline_value_list)

            ori_ECG_list = []
            for i in range(12):
                ori_ECG_list.append(torch.from_numpy(ECG_leadList[i][ECG_ll:ECG_rr]))
                # if type(ECG_leadList) == "np.ndarray":
                #     ori_ECG_list.append(torch.from_numpy(ECG_leadList[i][ECG_ll:ECG_rr]))
                # else:
                #     ori_ECG_list.append(ECG_leadList[i][ECG_ll:ECG_rr])

        elif beat_gen_normalize_mode == "ECG_normal":
            # 对ECG整体进行归一化
            pass

        gen_data_x = self.beat_de_crop(gen_data_x, beat_padding_index)

        gen_baseline_value_list = self.get_baseline_value(gen_data_x)
        # print("gen_baseline_value_list: ", gen_baseline_value_list)

        gen_ECG_list = self.beats_connect_to_ECG(gen_data_x)

        if de_baseline:
            # 统一减去基线数值
            for i in range(12):
                for j in range(ori_ECG_list[i].shape[0]):
                    ori_ECG_list[i][j] -= ori_baseline_value_list[i]
                    gen_ECG_list[i][j] -= gen_baseline_value_list[i]
                # print(ori_ECG_list[i].shape)
                # print(gen_ECG_list[i].shape)

        if isShow:

            plt.figure(figsize=(10, 15))
            title = "gen ECG"
            for i in range(12):
                # plt.ylim(-1, 1)
                plt.subplot(12, 1, i + 1)
                plt.plot(gen_ECG_list[i])
                # plt.title("lead:" + str(i + 1))

            plt.suptitle(title)
            plt.show()

            for i in range(12):
                if True:
                    show_heat_from_beat(ori_ECG_list[i].view(1, -1), gen_ECG_list[i].view(1, -1), isShow=True)

            for i in range(12):
                # print(gen_ECG_list[i].shape)
                plt.figure(figsize=(12, 4))
                plt.subplot(2, 1, 1)
                print("lead{}  ori ECG length:{}".format(i + 1, len(ori_ECG_list[i])),
                      "gen ECG length:{}".format(len(gen_ECG_list[i])))
                plt.plot(ori_ECG_list[i])
                plt.title("ori ECG lead: {} {}".format(str(i + 1), beat_gen_normalize_mode))

                plt.subplot(2, 1, 2)
                plt.plot(gen_ECG_list[i])
                plt.title("gen ECG lead:" + str(i + 1))
                plt.show()

        return ori_ECG_list, gen_ECG_list, label, len(beat_split_index), ori_baseline_value_list, gen_baseline_value_list

    def ECG_split_to_beats(self, ECG_leadList, label, isShow=False, beat_gen_normalize_mode="no_normal"):
        # 记录每个导联提取出来的beat数量
        beat_cnt = []
        data_x = []
        data_y = []

        beat_split_index, beat_padding_index, ECG_ll, ECG_rr = get_index_from_recording_12_lead(ECG_leadList)

        for i in range(12):
            ECG_list_x = []
            ECG_list_y = []
            tmp_x = get_beat_from_recording_by_index(ECG_leadList[i], beat_split_index, isShow=False)

            beat_cnt.append(len(tmp_x))
            for j in range(len(tmp_x)):
                # 添加导联label
                ECG_list_x.append(tmp_x[j])
                ECG_list_y.append([label, i])

            data_x.append(np.array(ECG_list_x).reshape((-1, self.opt.nc, self.opt.beat_size)))
            data_y.append(np.array(ECG_list_y))

        for i in range(12):
            if beat_gen_normalize_mode == "Beat_normal":
                for j in range(data_x[i].shape[0]):
                    for k in range(self.opt.nc):
                        data_x[i][j][k] = self.normalize(data_x[i][j][k])

            X = torch.from_numpy(data_x[i])
            Y = torch.from_numpy(data_y[i])

            X = torch.as_tensor(X, dtype=torch.float32)
            Y = torch.as_tensor(Y, dtype=torch.long).t()
            data_x[i] = X
            data_y[i] = Y

        return data_x, data_y, beat_split_index, beat_padding_index, ECG_ll, ECG_rr

    def gen_beat(self, data_x, data_y, isShow=False, isLoss=False):
        gen_data_x = []

        tmp_G_label_loss = 0.0
        tmp_G_gen_loss = 0.0
        tmp_G_feature_loss = 0.0

        print(data_x[0].shape)
        beat_num_lead = data_x[0].shape[0]
        tmp_real_labels = torch.ones(beat_num_lead, 1, device=self.device)
        tmp_fake_labels = torch.zeros(beat_num_lead, 1, device=self.device)

        for i in range(12):
            X = data_x[i].to(self.device)
            Y = data_y[i].to(self.device)

            lead_label = self.lead_label_1hots[Y[1]]
            if self.opt.model_mode == "CGAN":
                fake_imgs, G_fake_feature = self.G(X, lead_label)
            elif self.opt.model_mode == "CGanomaly":
                fake_imgs, G_fake_feature, G_encoder2_feature = self.G(X, lead_label)
            else:
                fake_imgs, G_fake_feature = self.G(X)

            ori_beats = X.view(-1, self.opt.beat_size).detach().cpu()
            gen_beats = fake_imgs.view(-1, self.opt.beat_size).detach().cpu()

            gen_data_x.append(gen_beats)

        if isShow:
            for i in range(len(gen_data_x)):
                beats_show(data_x[i].view(-1, self.opt.beat_size), title="ori beat lead{}".format(i))
                beats_show(gen_data_x[i], title="gen beat lead{}, length:{}".format(i, len(gen_data_x[i])))

        return gen_data_x

    def beats_connect_to_ECG(self, beatlist):
        ECG_leadList = []

        for beats in beatlist:
            # 对12导联的Beats进行拼接
            tmp_ECG = beats[0]
            for i in range(len(beats)):
                if i == 0:
                    continue
                else:
                    tmp_ECG = torch.cat((tmp_ECG, beats[i]), 0)

            ECG_leadList.append(tmp_ECG)

        return ECG_leadList

    def beat_de_crop(self, gen_beatlist, beat_padding_index):
        # 根据beat_padding_index对生成的长度为500的beat进行反裁剪
        new_beatlist = []
        for i in range(12):
            beats = gen_beatlist[i]
            tmp_beats = []
            for j in range(len(beats)):
                ll, rr = beat_padding_index[j]
                if ll >= 0 and rr >= 0:
                    if rr == 0:
                        beat = beats[j][ll:]
                    else:
                        beat = beats[j][ll:-rr]
                elif ll >= 0 and rr < 0:
                    beat = beats[j][ll:]
                    beat_np = beat.numpy()
                    for k in range(-rr):
                        beat_np = np.insert(beat_np, -1, beat_np[-1])
                    beat = torch.from_numpy(beat_np)
                elif ll < 0 and rr >= 0:
                    beat = beats[j][:-rr]
                    beat_np = beat.numpy()
                    for k in range(-ll):
                        beat_np = np.insert(beat_np, 0, beat_np[0])
                    beat = torch.from_numpy(beat_np)
                else:
                    beat_np = beats[j].numpy()
                    for k in range(-ll):
                        beat_np = np.insert(beat_np, 0, beat_np[0])
                    for k in range(-rr):
                        beat_np = np.insert(beat_np, -1, beat_np[-1])
                    beat = torch.from_numpy(beat_np)

                tmp_beats.append(beat)

            new_beatlist.append(tmp_beats)

        return new_beatlist

    def get_baseline_value(self, data, split_index=None):
        baseline_value_list = []
        if split_index is None:

            for i in range(12):
                tmp_baseline_value = 0
                for j in range(len(data[i])):
                    # print(type(data[i][j]), data[i][j].shape)
                    if data[i][j].shape[0] == 1:
                        try:
                            tmp_baseline_value += (data[i][j][0][0].item() + data[i][j][0][-1].item())
                        except:
                            tmp_baseline_value += (data[i][j][0].item() + data[i][j][-1].item())
                    else:
                        tmp_baseline_value += (data[i][j][0].item() + data[i][j][-1].item())
                tmp_baseline_value = tmp_baseline_value / (2 * len(data[i]))
                # print("tmp: ", tmp_baseline_value)
                baseline_value_list.append(tmp_baseline_value)
        else:
            for i in range(12):
                tmp_baseline_value = 0
                for j in range(len(split_index)):
                    tmp_baseline_value += (data[i][split_index[j][0]] + data[i][split_index[j][1]])
                tmp_baseline_value = tmp_baseline_value / (2 * len(split_index))
                # print("tmp: ", tmp_baseline_value)
                baseline_value_list.append(tmp_baseline_value)

        return baseline_value_list
