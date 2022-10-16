class Options(object):
    def __init__(self):
        self.dataset_folder = None

        self.normal_data_TrainSetBeat_GAN_X = None
        self.normal_data_TrainSetBeat_GAN_Y = None
        self.abnormal_data_TrainSetBeat_GAN_X = None
        self.abnormal_data_TrainSetBeat_GAN_Y = None

        self.log_save_path = None

        self.isNormalization = False
        self.preprocess = "max_min"

        self.random_state = 2022
        self.ngpu = 2

        self.model_mode = "CGAN"
        self.model_index = "100"

        self.beat_channels = 1
        self.nc = 1
        self.nz = 64
        self.ndf = 32
        self.ngf = 32

        self.beat_size = 500
        self.num_leads = 12
        self.ECG_matrix_length = 5000

        self.batch_size = 128
        self.num_epochs = 150
        self.lr = 0.001

        # 设置G的惩罚轮数
        self.n_critic = 2
        self.clamp_num = 0.01
        self.dropout_ratio = 0.3

        self.G_label_loss_weight = 1
        self.G_gen_loss_weight = 1
        self.G_feature_loss_weight = 1

        self.dataset = "CPSC"
        self.experiment_date = "1010"
        self.isSave_val_loss = False