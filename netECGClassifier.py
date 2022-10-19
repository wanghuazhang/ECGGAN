import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # torch.nn.init.xavier_uniform_(m.weight.data)
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data)
        # torch.nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.01)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        # torch.nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.01)


# channel attention: C×H×W ------> C×1×1
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        out_channel = channel // ratio
        if out_channel == 0:
            out_channel = 1
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, out_channel, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channel, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


# spatial attention: C×H×W ------> 1×H×W
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ECG_Classifier_CNN_CBAM_5000(nn.Module):
    def __init__(self, opt):
        super(ECG_Classifier_CNN_CBAM_5000, self).__init__()
        self.opt = opt

        self.conv = nn.Sequential(
            # input 1 * 12 * 5000
            nn.Conv2d(self.opt.input_channel, 8, (2, 40), (1, 4), bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            # 8 * 11 * 1241
            nn.Conv2d(8, 32, (3, 37), (1, 4), bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 32 * 9 * 302
            nn.Conv2d(32, 128, (4, 26), (1, 4), bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 128 * 6 * 70
            nn.Conv2d(128, 256, (3, 16), (1, 2), bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 256 * 4 * 28
            nn.Conv2d(256, 512, (3, 4), (1, 2), bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 * 2 * 13
        )

        self.cbam = CBAM(channel=512)

        self.linear = nn.Sequential(
            nn.Linear(
                in_features=512 * 2 * 13,
                out_features=2
            ),
            nn.Softmax()
        )

    def forward(self, x):
        # x = self.cbam(x)
        x = self.conv(x)
        if self.opt.is_CBAM:
            x = self.cbam(x)
        x = x.view(-1, 512 * 2 * 13)
        y = self.linear(x)

        return y


class ECG_Classifier_CNN_CBAM_5000_multi_filter(nn.Module):
    def __init__(self, opt):
        super(ECG_Classifier_CNN_CBAM_5000_multi_filter, self).__init__()
        self.opt = opt

        self.lead_filter_size = [1, 2, 3, 4, 6, 8, 12]
        # self.lead_filter_size = [1, 2, 3, 4, 6]
        self.conv_list = nn.ModuleList([nn.Conv2d(self.opt.input_channel, 8, (size, 40), (1, 4), bias=False) for size in self.lead_filter_size])
        self.Dropout_multi = nn.Dropout(self.opt.dropout_ratio)
        self.BN_multi = nn.BatchNorm2d(8)
        self.Relu = nn.ReLU(True)


        if len(self.lead_filter_size) == 7:
            self.conv2 = nn.Sequential(
                # input 1 * 12 * 5000
                # nn.Conv2d(1, 8, (2, 40), (1, 4), bias=False),
                # nn.Dropout(self.opt.dropout_ratio),
                # nn.BatchNorm2d(8),
                # nn.ReLU(True),

                # 8 * 55 * 1241
                nn.Conv2d(8, 32, (3, 37), (2, 4), bias=False),
                nn.Dropout(self.opt.dropout_ratio),
                nn.BatchNorm2d(32),
                nn.ReLU(True),

                # 32 * 27 * 302
                nn.Conv2d(32, 128, (3, 26), (3, 4), bias=False),
                nn.Dropout(self.opt.dropout_ratio),
                nn.BatchNorm2d(128),
                nn.ReLU(True),

                # 128 * 9 * 70
                nn.Conv2d(128, 256, (3, 16), (2, 2), bias=False),
                nn.Dropout(self.opt.dropout_ratio),
                nn.BatchNorm2d(256),
                nn.ReLU(True),

                # 256 * 4 * 28
                nn.Conv2d(256, 512, (3, 4), (1, 2), bias=False),
                nn.Dropout(self.opt.dropout_ratio),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # 512 * 2 * 13
            )
        elif len(self.lead_filter_size) == 5:
            self.conv2 = nn.Sequential(
                # 8 * 49 * 1241
                nn.Conv2d(8, 32, (3, 37), (2, 4), bias=False),
                nn.Dropout(self.opt.dropout_ratio),
                nn.BatchNorm2d(32),
                nn.ReLU(True),

                # 32 * 24 * 302
                nn.Conv2d(32, 128, (3, 26), (3, 4), bias=False),
                nn.Dropout(self.opt.dropout_ratio),
                nn.BatchNorm2d(128),
                nn.ReLU(True),

                # 128 * 8 * 70
                nn.Conv2d(128, 256, (2, 16), (2, 2), bias=False),
                nn.Dropout(self.opt.dropout_ratio),
                nn.BatchNorm2d(256),
                nn.ReLU(True),

                # 256 * 4 * 28
                nn.Conv2d(256, 512, (3, 4), (1, 2), bias=False),
                nn.Dropout(self.opt.dropout_ratio),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # 512 * 2 * 13
            )

        self.cbam = CBAM(channel=512)

        self.linear = nn.Sequential(
            nn.Linear(
                in_features=512 * 2 * 13,
                out_features=2
            ),
            nn.Softmax()
        )

    def forward(self, x):
        multi_lead_outputs = []
        for i, conv in enumerate(self.conv_list):
            tmp_conv = conv(x)
            tmp_conv = self.Dropout_multi(tmp_conv)
            tmp_conv = self.BN_multi(tmp_conv)
            tmp_conv = self.Relu(tmp_conv)
            multi_lead_outputs.append(tmp_conv)

        multi_conv = torch.cat(multi_lead_outputs, 2)

        x = self.conv2(multi_conv)
        if self.opt.is_CBAM:
            x = self.cbam(x)
        x = x.view(-1, 512 * 2 * 13)
        y = self.linear(x)

        return y


class ECG_Classifier_CNN_CBAM_5000_simple_3CNN(nn.Module):
    def __init__(self, opt):
        super(ECG_Classifier_CNN_CBAM_5000_simple_3CNN, self).__init__()
        self.opt = opt

        self.conv = nn.Sequential(
            # input 1 * 12 * 5000
            nn.Conv2d(1, 8, (3, 40), (1, 4), bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            # 8 * 10 * 1241
            nn.Conv2d(8, 32, (2, 37), (2, 4), bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 32 * 5 * 302
            nn.Conv2d(32, 128, (4, 22), (1, 8), bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 128 * 2 * 36
        )

        self.cbam = CBAM(channel=128)

        self.linear = nn.Sequential(
            nn.Linear(
                in_features=128 * 2 * 36,
                out_features=2
            ),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv(x)
        if self.opt.is_CBAM:
            x = self.cbam(x)
        x = x.view(-1, 128 * 2 * 36)
        y = self.linear(x)

        return y


class ECG_Classifier_CNN_CBAM_5000_simple_3CNN_multi_filter(nn.Module):
    def __init__(self, opt):
        super(ECG_Classifier_CNN_CBAM_5000_simple_3CNN_multi_filter, self).__init__()
        self.opt = opt

        self.lead_filter_size = [1, 2, 3, 4, 6, 8, 12]
        # self.lead_filter_size = [1, 2, 3, 4, 6]
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(self.opt.input_channel, 8, (size, 40), (1, 4), bias=False) for size in self.lead_filter_size])
        self.Dropout_multi = nn.Dropout(self.opt.dropout_ratio)
        self.BN_multi = nn.BatchNorm2d(8)
        self.Relu = nn.ReLU(True)

        self.conv2 = nn.Sequential(
            # 8 * 55 * 1241
            nn.Conv2d(8, 32, (7, 37), (4, 4), bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 32 * 13 * 302
            nn.Conv2d(32, 128, (5, 22), (4, 8), bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 128 * 3 * 36
        )

        self.cbam = CBAM(channel=128)

        self.linear = nn.Sequential(
            nn.Linear(
                in_features=128 * 3 * 36,
                out_features=2
            ),
            nn.Softmax()
        )

    def forward(self, x):
        multi_lead_outputs = []
        for i, conv in enumerate(self.conv_list):
            # input: 1 * 12 * 5000
            # print(i, conv)
            tmp_conv = conv(x)
            tmp_conv = self.Dropout_multi(tmp_conv)
            tmp_conv = self.BN_multi(tmp_conv)
            tmp_conv = self.Relu(tmp_conv)
            # output: 8 * (12 - filter + 1) * 1241
            # print(i, tmp_conv.size())
            multi_lead_outputs.append(tmp_conv)

        multi_conv = torch.cat(multi_lead_outputs, 2)
        # 8 * 55 * 1241

        x = self.conv2(multi_conv)
        if self.opt.is_CBAM:
            x = self.cbam(x)
        x = x.view(-1, 128 * 3 * 36)
        y = self.linear(x)

        return y