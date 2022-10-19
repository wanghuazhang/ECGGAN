class ECG_Options(object):
    def __init__(self):
        self.ngpu = 1
        self.ECG_model_path = ""

        self.random_state = 2021

        # 训练次数
        self.num_epochs = 200

        self.batch_size = 16
        self.lr = 0.01

        self.optimizer_name = "SGD"
        self.criterion_name = "CrossEntropyLoss"

        self.LeakyReLU_slope = 0.2
        self.dropout_ratio = 0.3

        self.momentum = 0.85

        self.index = 1

        self.ECG_batch_size = 32
        self.ECG_matrix_length = 5000

        self.is_multi_filter = False
        self.is_channel_filter = False
        self.isNormalization = False
        self.normal_mode = "no_normal"
        self.net_mode = "5CNN"
        self.is_CBAM = True

        self.beat_gen_normalize_mode = "no_normal"

        self.input_channel = 1

        self.dataset = "CPSC"

        self.val_fold = 9
        self.test_fold = 10

        self.cnt = 0
        self.experiment_date = "1010"





