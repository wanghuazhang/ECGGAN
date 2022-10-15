class Options(object):
    def __init__(self):
        self.dataset_folder = None
        self.abnormal_data_TrainSetBeat_GAN_X = "/data1/wanghuazhang/graduation/GANECG/dataset/TrainSetBeat_GAN/abnormal_beat_lead_list_X.npy"
        self.abnormal_data_TrainSetBeat_GAN_Y = "/data1/wanghuazhang/graduation/GANECG/dataset/TrainSetBeat_GAN/abnormal_beat_lead_list_Y.npy"

        self.normal_data_TrainSetBeat_GAN_X = "/data1/wanghuazhang/graduation/GANECG/dataset/ECG_traindata/CPSC_AIWIN_normal_beat_Generate_lead_list_X.npy"
        self.normal_data_TrainSetBeat_GAN_Y = "/data1/wanghuazhang/graduation/GANECG/dataset/ECG_traindata/CPSC_AIWIN_normal_beat_Generate_lead_list_Y.npy"

        self.log_save_path = None

        self.isNormalization = False
        self.preprocess = "max_min"

        self.random_state = 2022
        self.ngpu = 2

        self.model_mode = "CGAN"
        self.model_index = "100"

        self.beat_channels = 1

        self.beat_size = 500
        self.num_leads = 12
        self.ECG_matrix_length = 5000

        self.batch_size = 128
        self.num_epochs = 102
        self.lr = 0.001

        # 设置G的惩罚轮数
        self.n_critic = 2
        self.clamp_num = 0.01
        self.dropout_ratio = 0.3

        self.G_label_loss_weight = 1
        self.G_gen_loss_weight = 1
        self.G_feature_loss_weight = 1

        self.dataset = "CPSC"