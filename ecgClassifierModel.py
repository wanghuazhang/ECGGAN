import os
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn import metrics
from torch.utils.data import dataloader, TensorDataset

import matplotlib.pyplot as plt

from netECGClassifier import ECG_Classifier_CNN_CBAM_5000_simple_3CNN, ECG_Classifier_CNN_CBAM_5000, \
    ECG_Classifier_CNN_CBAM_5000_simple_3CNN_multi_filter, ECG_Classifier_CNN_CBAM_5000_multi_filter, weights_init
from preProcess import ECG_crop


class ecgClassifier():
    def __init__(self, opt, mode=None, ECG_gen_model=None, CNN_path=None, Generator_model_path=None):
        self.opt = opt
        self.mode = mode
        self.ECG_gen_model = ECG_gen_model
        self.CNN_path = CNN_path
        self.Generator_model_path = Generator_model_path

        self.Generator_model_name = Generator_model_path.split('/')[-1].split('.')[0]
        self.X_path = None
        self.Y_path = None

        self.save_train_model = "iter"

        self.f_train_log = None

        self.dataset = self.opt.dataset

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.best_auc = 0
        self.prec = 0
        self.rec = 0
        self.best_f1 = 0
        self.best_thr = 0
        self.best_epoch = 0

        self.best_auc_test = 0
        self.best_epoch_test = 0

        self.best_auc_train = 0
        self.best_epoch_train = 0

        self.auc_list_train = []
        self.auc_list_val = []
        self.auc_list_test = []

        self.F1_list_train = []
        self.F1_list_val = []
        self.F1_list_test = []

        self.data_loader = self.load_ECG_matrix_data_from_folder()

        if not self.opt.is_multi_filter:
            # only CNN
            if self.opt.net_mode == "3CNN":
                self.ECG_model = ECG_Classifier_CNN_CBAM_5000_simple_3CNN(self.opt)
            elif self.opt.net_mode == "5CNN":
                self.ECG_model = ECG_Classifier_CNN_CBAM_5000(self.opt)
        elif self.opt.is_multi_filter:
            # CNN + multi filter
            if self.opt.net_mode == "3CNN":
                self.ECG_model = ECG_Classifier_CNN_CBAM_5000_simple_3CNN_multi_filter(self.opt)
            elif self.opt.net_mode == "5CNN":
                self.ECG_model = ECG_Classifier_CNN_CBAM_5000_multi_filter(self.opt)

        self.create_log_folder()

        if (self.device.type == 'cuda') and (self.opt.ngpu > 1):
            self.ECG_model = nn.DataParallel(self.ECG_model, list(range(self.opt.ngpu)))

        self.ECG_model.to(self.device)
        self.ECG_model.apply(weights_init)

        if self.opt.criterion_name == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        elif self.opt.criterion_name == "BCELoss":
            self.criterion = nn.BCELoss()
        elif self.opt.criterion_name == "BCEWithLogitsLoss":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.opt.criterion_name == "MSELoss":
            self.criterion = nn.MSELoss()

        if self.opt.optimizer_name == "SGD":
            self.SGD_optimizer = optim.SGD(self.ECG_model.parameters(), lr=self.opt.lr, momentum=self.opt.momentum)
            self.optimizer = self.SGD_optimizer
        elif self.opt.optimizer_name == "ASGD":
            self.ASGN_optimizer = optim.ASGD(self.ECG_model.parameters(), lr=self.opt.lr, lambd=0.0001, alpha=0.75,
                                             t0=1000000.0, weight_decay=0)
            self.optimizer = self.ASGN_optimizer
        elif self.opt.optimizer_name == "RMSprop":
            self.RMSprop_optimizer = optim.RMSprop(self.ECG_model.parameters(), lr=self.opt.lr, alpha=0.99,
                                                   eps=1e-08,
                                                   weight_decay=0, momentum=0, centered=False)
            self.optimizer = self.RMSprop_optimizer
        elif self.opt.optimizer_name == "Adam":
            self.Adam_optimizer = optim.Adam(self.ECG_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.999),
                                             eps=1e-08, weight_decay=0, amsgrad=False)
            self.optimizer = self.Adam_optimizer

        # train_time是每一个epoch训练时间
        self.train_time = []

    def normalize(self, seq):
        if self.opt.isNormalization:
            if np.max(seq) == np.min(seq):
                return seq
            return 2 * (seq - np.min(seq)) / (np.max(seq) - np.min(seq)) - 1
        else:
            return seq

    def normalize_12_lead(self, seq):
        if self.opt.normal_mode == "one_lead":
            for i in range(12):
                seq[i] = self.normalize(seq[i])
        elif self.opt.normal_mode == "all_lead":
            seq = self.normalize(seq)
        elif self.opt.normal_mode == "no_normal":
            return seq
        return seq

    def load_ECG_matrix_data_from_folder(self):
        # 读取指定的train、val、test数据集
        self.data_save_folder = self.Generator_model_path[: 0 - len(self.Generator_model_path.split('/')[-1])]
        print(self.data_save_folder)

        self.train_X_path = os.path.join(self.data_save_folder,
                                         self.dataset + "_train_ECG_matrix_X_path_5000_" + self.Generator_model_name + "_gen_{}.npy".format(
                                             self.opt.beat_gen_normalize_mode))
        self.train_Y_path = os.path.join(self.data_save_folder,
                                         self.dataset + "_train_ECG_matrix_Y_path_5000_" + self.Generator_model_name + "_gen_{}.npy".format(
                                             self.opt.beat_gen_normalize_mode))

        self.val_X_path = os.path.join(self.data_save_folder,
                                       self.dataset + "_val_ECG_matrix_X_path_5000_" + self.Generator_model_name + "_gen_{}.npy".format(
                                           self.opt.beat_gen_normalize_mode))
        self.val_Y_path = os.path.join(self.data_save_folder,
                                       self.dataset + "_val_ECG_matrix_Y_path_5000_" + self.Generator_model_name + "_gen_{}.npy".format(
                                           self.opt.beat_gen_normalize_mode))

        self.test_X_path = os.path.join(self.data_save_folder,
                                        self.dataset + "_test_ECG_matrix_X_path_5000_" + self.Generator_model_name + "_gen_{}.npy".format(
                                            self.opt.beat_gen_normalize_mode))
        self.test_Y_path = os.path.join(self.data_save_folder,
                                        self.dataset + "_test_ECG_matrix_Y_path_5000_" + self.Generator_model_name + "_gen_{}.npy".format(
                                            self.opt.beat_gen_normalize_mode))

        print(self.train_X_path)

        if not os.path.exists(self.train_X_path):
            ECG_matrix_X_train, ECG_matrix_Y_train = self.get_ECG_matrix_data_based_Generator_model_from_numpy_data(
                self.train_X_path,
                self.train_Y_path,
                dataset='train')
        else:
            ECG_matrix_X_train = np.load(self.train_X_path)
            ECG_matrix_Y_train = np.load(self.train_Y_path)

        if not os.path.exists(self.val_X_path):
            ECG_matrix_X_val, ECG_matrix_Y_val = self.get_ECG_matrix_data_based_Generator_model_from_numpy_data(
                self.val_X_path,
                self.val_Y_path,
                dataset='val')
        else:
            ECG_matrix_X_val = np.load(self.val_X_path)
            ECG_matrix_Y_val = np.load(self.val_Y_path)

        if not os.path.exists(self.test_X_path):
            ECG_matrix_X_test, ECG_matrix_Y_test = self.get_ECG_matrix_data_based_Generator_model_from_numpy_data(
                self.test_X_path,
                self.test_Y_path,
                dataset='test')
        else:
            ECG_matrix_X_test = np.load(self.test_X_path)
            ECG_matrix_Y_test = np.load(self.test_Y_path)

        print("Train size: ", ECG_matrix_X_train.shape)
        print("Val size: ", ECG_matrix_X_val.shape)
        print("Test size: ", ECG_matrix_X_test.shape)

        print("normalize : {}".format(self.opt.normal_mode))
        if self.opt.normal_mode != "no_normal":
            for i in range(ECG_matrix_X_train.shape[0]):
                ECG_matrix_X_train[i] = self.normalize_12_lead(ECG_matrix_X_train[i])
            for i in range(ECG_matrix_X_val.shape[0]):
                ECG_matrix_X_val[i] = self.normalize_12_lead(ECG_matrix_X_val[i])
            for i in range(ECG_matrix_X_test.shape[0]):
                ECG_matrix_X_test[i] = self.normalize_12_lead(ECG_matrix_X_test[i])

        train_dataset = TensorDataset(torch.from_numpy(ECG_matrix_X_train), torch.from_numpy(ECG_matrix_Y_train))
        val_dataset = TensorDataset(torch.from_numpy(ECG_matrix_X_val), torch.from_numpy(ECG_matrix_Y_val))
        test_dataset = TensorDataset(torch.from_numpy(ECG_matrix_X_test), torch.from_numpy(ECG_matrix_Y_test))

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

    def get_ECG_matrix_data_based_Generator_model_from_numpy_data(self, X_path, Y_path, dataset='train'):
        # 针对10折对比实验，直接从封装保存好的numpy data中读取X、Y，只需要指定dataset和fold num
        data_folder = "./experiment/{}/dataset_val={}_test={}".format(
            self.opt.dataset, self.opt.val_fold, self.opt.test_fold)

        raw_data_X_path = os.path.join(data_folder, "raw_data_X_{}.npy".format(dataset))
        raw_data_Y_path = os.path.join(data_folder, "raw_data_Y_{}.npy".format(dataset))
        print("ecgClassifierModer raw data x path: ", raw_data_X_path)

        raw_data_X = np.load(raw_data_X_path)
        raw_data_Y = np.load(raw_data_Y_path)

        error_matrix_list = []
        label_list = []

        for cnt in range(raw_data_X.shape[0]):
            print(cnt, raw_data_Y[cnt])
            try:
                ECG_leadList = raw_data_X[cnt]
                if ECG_leadList.shape[1] != 5000:
                    # 这里是因为切割的Beat要“掐头去尾”，所以相当于先往两侧延申一下
                    ECG_leadList = ECG_crop(ECG_leadList, avg_length=self.opt.ECG_matrix_length + 1000)

                data_tmp = [ECG_leadList, raw_data_Y[cnt][0], raw_data_Y[cnt][1]]

                ori_ECG_list, gen_ECG_list, label, beat_cnt, ori_baseline_value_list, gen_baseline_value_list = self.ECG_gen_model.gen_ECG_item(
                    data_tmp, de_baseline=True, beat_gen_normalize_mode=self.opt.beat_gen_normalize_mode)

                ori_ECG_list_drop = ECG_crop(ori_ECG_list, avg_length=self.opt.ECG_matrix_length)
                gen_ECG_list_crop = ECG_crop(gen_ECG_list, avg_length=self.opt.ECG_matrix_length)

                error_matrix = []

                for j in range(12):
                    ori_ECG = ori_ECG_list_drop[j]
                    gen_ECG = gen_ECG_list_crop[j]
                    lead_matrix = gen_ECG - ori_ECG
                    error_matrix.append(lead_matrix)

                error_matrix_list.append(np.array(error_matrix))
                label_list.append(np.array([raw_data_Y[cnt][0], raw_data_Y[cnt][1], raw_data_Y[cnt][2]]))
            except:
                print("error!!! ****** ", cnt, raw_data_Y[cnt])

            ECG_matrix_X = np.array(error_matrix_list)
            print("ECG matrix X shape: ", ECG_matrix_X.shape)
            ECG_matrix_Y = np.array(label_list)
            print("ECG matrix Y shape: ", ECG_matrix_Y.shape)
            print(ECG_matrix_Y)

        np.save(X_path, ECG_matrix_X)
        np.save(Y_path, ECG_matrix_Y)
        return ECG_matrix_X, ECG_matrix_Y

    def create_log_folder(self):

        log_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
        print("Time:", log_time)
        log_file_name = "{}/{}_Gen_{}_{}_Matrix_{}_LR={}_Drop={}_Momen={}_{}".format(
            self.opt.experiment_date,
            self.opt.cnt,
            self.opt.beat_gen_normalize_mode,
            self.opt.net_mode,
            self.opt.normal_mode,
            self.opt.lr,
            self.opt.dropout_ratio,
            self.opt.momentum,
            self.opt.index)

        self.log_folder = os.path.join(self.data_save_folder, log_file_name)
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)  # makedirs 创建文件时如果路径不存在会创建这个路径
            print("create folde: ", self.log_folder)

        # 保存本次模型训练的基本参数
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
        self.f_train_log.write('\n')

        print("Model Structure: ", self.ECG_model)
        self.f_train_log.write("Model Structure: \n")
        self.f_train_log.write(str(self.ECG_model))
        self.f_train_log.write('\n')

        para_content = self.get_parameter_number(self.ECG_model)

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        para_content = 'Total: {}, Trainable: {}'.format(total_num, trainable_num)
        print(para_content)
        self.f_train_log.write(para_content)
        self.f_train_log.write('\n')
        return para_content

    def save_model(self, title=None):
        ECG_model_path = os.path.join(self.log_folder, title + "ECG_Classifier_CNN.pkl")
        torch.save(self.ECG_model, ECG_model_path)
        self.opt.ECG_model_path = ECG_model_path

    def train_ECG_Matrix_Classifier_Model(self, isSave=True):
        print("Starting Training Loop...")
        start_time = time.time()
        for epoch in range(self.opt.num_epochs):
            epoch_start_time = time.time()
            running_loss = 0.0

            self.ECG_model.train()
            torch.set_grad_enabled(True)
            for i, data in enumerate(self.data_loader['train']):
                X = torch.as_tensor(data[0], dtype=torch.float32)
                X = X.view(-1, 1, 12, self.opt.ECG_matrix_length)

                tmp_Y = torch.index_select(data[1], dim=1, index=torch.tensor([0])).view(-1)
                Y = torch.as_tensor(tmp_Y, dtype=torch.long)

                X = X.to(self.device)
                Y = Y.to(self.device)

                outputs = self.ECG_model(X)

                ECG_loss = self.criterion(outputs, Y)

                self.optimizer.zero_grad()
                ECG_loss.backward()

                self.optimizer.step()

                running_loss += ECG_loss

            self.train_time.append(time.time() - epoch_start_time)
            time_log_content = "Epoch {}: training time: {:.2f}, Total time: {:.2f}, avg time: {:.2f}" \
                .format(epoch, self.train_time[epoch], sum(self.train_time), sum(self.train_time) / (epoch + 1))
            self.f_train_log.write(time_log_content)
            self.f_train_log.write('\n')
            loss_log_content = 'Epoch {}: ECG_loss: {}'.format(epoch, running_loss / (i + 1))
            self.f_train_log.write(loss_log_content)
            self.f_train_log.write('\n')

            print(time_log_content)
            print(loss_log_content)

            # get auc、th、f1 for each epoch
            auc, prec, rec, best_f1, best_thr = self.validate()
            self.auc_list_val.append(auc)
            self.F1_list_val.append([best_f1, prec, rec])
            if auc - self.best_auc > 0.0005:
                self.best_auc = auc
                self.prec = prec
                self.rec = rec
                self.best_f1 = best_f1
                self.best_thr = best_thr
                self.best_epoch = epoch
                if isSave:
                    model_title = "train_auc_{:.4f}_".format(self.best_auc)
                    if self.save_train_model == "iter":
                        self.save_model(model_title)
                    else:
                        self.save_model()

            auc_log_content = "Epoch {}: auc{:.4f} precision {:.4f} recall {:.4f} best_f1 {:.4f} best_threshold {:.4f}, best_auc={:.4f} in epoch{}".format(
                epoch, auc, prec, rec, best_f1, best_thr, self.best_auc, self.best_epoch)

            self.f_train_log.write(auc_log_content)
            self.f_train_log.write('\n')

            print(auc_log_content)

            auc, prec, rec, best_f1, best_thr = self.validate('test')
            self.auc_list_test.append(auc)
            self.F1_list_test.append([best_f1, prec, rec])
            if auc - self.best_auc_test > 0.0005:
                self.best_auc_test = auc
                self.best_epoch_test = epoch
                if isSave:
                    model_title = "test_auc_{:.4f}_".format(self.best_auc_test)
                    if self.save_train_model == "iter":
                        self.save_model(model_title)
                    else:
                        self.save_model(model_title)

            auc_log_content = "***Test*** Epoch {} : auc{:.4f} precision {:.4f} recall {:.4f} best_f1 {:.4f} best_threshold {:.4f}, best_auc={:.4f} in epoch{}".format(
                epoch, auc, prec, rec, best_f1, best_thr, self.best_auc_test, self.best_epoch_test)

            self.f_train_log.write(auc_log_content)
            self.f_train_log.write('\n')

            print(auc_log_content)

            auc, prec, rec, best_f1, best_thr = self.validate('train')
            self.auc_list_train.append(auc)
            self.F1_list_train.append([best_f1, prec, rec])

            auc_log_content = "***Train*** Epoch {} : auc{:.4f} precision {:.4f} recall {:.4f} best_f1 {:.4f} best_threshold {:.4f}, best_auc={:.4f} in epoch{}".format(
                epoch, auc, prec, rec, best_f1, best_thr, self.best_auc_train, self.best_epoch_train)

            self.f_train_log.write(auc_log_content)
            self.f_train_log.write('\n')

            print(auc_log_content)

            print("Epoch {} LR: {}".format(epoch, self.optimizer.state_dict()['param_groups'][0]['lr']))

        print("End Training.")

        for i in range(len(self.auc_list_val)):
            tmp_log_content = "epoch {}: val auc {:.4f}, test auc {:.4f}, train auc {:.4f}\n".format(i, self.auc_list_val[i], self.auc_list_test[i], self.auc_list_train[i])
            self.f_train_log.write(tmp_log_content)

        # 从大到小排序，并且记录下标
        sorted_nums = sorted(enumerate(self.auc_list_val), key=lambda x: x[1], reverse=True)
        idx = [i[0] for i in sorted_nums]
        nums = [i[1] for i in sorted_nums]
        for item in idx:
            tmp_log_content = "Result epoch {}: val auc {:.4f}, test auc {:.4f}, train auc {:.4f}\n".format(item, self.auc_list_val[item], self.auc_list_test[item], self.auc_list_train[item])
            self.f_train_log.write(tmp_log_content)

        sorted_nums = sorted(enumerate(self.auc_list_test), key=lambda x: x[1], reverse=True)
        idx = [i[0] for i in sorted_nums]
        nums = [i[1] for i in sorted_nums]
        for item in idx:
            tmp_log_content = "sorted test auc epoch {}: {:.4f}\n".format(item, self.auc_list_test[item])
            self.f_train_log.write(tmp_log_content)

        sorted_nums = sorted(enumerate(self.auc_list_train), key=lambda x: x[1], reverse=True)
        idx = [i[0] for i in sorted_nums]
        nums = [i[1] for i in sorted_nums]
        for item in idx:
            tmp_log_content = "sorted train auc epoch {}: {:.4f}\n".format(item, self.auc_list_train[item])
            self.f_train_log.write(tmp_log_content)

        # 对F1从大到小排序，并且记录下标
        sorted_nums = sorted(enumerate(self.F1_list_val), key=lambda x: x[1], reverse=True)
        F1_idx = [i[0] for i in sorted_nums]
        nums = [i[1] for i in sorted_nums]
        for item in F1_idx:
            tmp_log_content = "F1 Result epoch {}: Val F1 {:.4f}, p {:.4f}, r {:.4f}, auc {:.4f}, sorted Test F1 {:.4f}, " \
                              "p {:.4f}, r {:.4f}, auc {:.4f}, Train F1 {:.4f}\n".format(item,
                                                                                         self.F1_list_val[item][0],
                                                                                         self.F1_list_val[item][1],
                                                                                         self.F1_list_val[item][2],
                                                                                         self.auc_list_val[item],
                                                                                         self.F1_list_test[item][0],
                                                                                         self.F1_list_test[item][1],
                                                                                         self.F1_list_test[item][2],
                                                                                         self.auc_list_test[item],
                                                                                         self.F1_list_train[
                                                                                             item][0])
            self.f_train_log.write(tmp_log_content)

        self.f_train_log.close()

        x_axix = list(range(self.opt.num_epochs))
        plt.ylim(0.5, 1)
        plt.title('AUC Result')
        plt.plot(x_axix, self.auc_list_train, color='green', label='train auc')
        plt.plot(x_axix, self.auc_list_val, color='red', label='val auc')
        plt.plot(x_axix, self.auc_list_test, color='blue', label='test auc')
        plt.legend()

        plt.xlabel('epoch')
        plt.ylabel('AUC')
        plt.savefig(os.path.join(self.log_folder, "AUC.jpg"))
        plt.show()
        plt.close()

        return F1_idx, self.F1_list_val, self.F1_list_test, self.F1_list_train, self.auc_list_val, self.auc_list_test, self.auc_list_train

    def validate(self, dataset='val'):
        y, y_pred = self.predict(self.data_loader[dataset])
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

        return auc, prec, rec, best_f1, best_thr

    def predict(self, data_loader):
        print("predicting")
        self.ECG_model.eval()
        torch.set_grad_enabled(False)
        with torch.no_grad():
            y_labels = torch.zeros(size=(len(data_loader.dataset),), dtype=torch.long, device=self.device)
            y_pred = torch.zeros(size=(len(data_loader.dataset),), dtype=torch.long, device=self.device)

            for i, data in enumerate(data_loader, 0):
                X = torch.as_tensor(data[0], dtype=torch.float32, device=self.device)
                X = X.view(-1, 1, 12, self.opt.ECG_matrix_length)

                tmp_Y = torch.index_select(data[1], dim=1, index=torch.tensor([0])).view(-1)
                Y = torch.as_tensor(tmp_Y, dtype=torch.long)

                outputs = self.ECG_model(X)

                y_labels[i * self.opt.batch_size: i * self.opt.batch_size + Y.shape[0]] = Y

                for index in range(Y.shape[0]):
                    # print(outputs[index])
                    if outputs[index][0] > outputs[index][1]:
                        # abnormal class
                        y_pred[i * self.opt.batch_size + index] = 0
                    else:
                        # normal class
                        y_pred[i * self.opt.batch_size + index] = 1

            y = y_labels.cpu().numpy()
            y_pred = y_pred.cpu().numpy()

        return y, y_pred