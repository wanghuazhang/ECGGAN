import torch
import torch.nn as nn


def weights_init(mod):
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data)
        # nn.init.kaiming_uniform_(mod.weight.data)

    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)


# 使用CGAN的encode、decode结构的判别器、生成器
class Encoder_CGAN(nn.Module):
    def __init__(self, opt):
        super(Encoder_CGAN, self).__init__()
        self.opt = opt
        self.main = nn.Sequential(
            # state size. (ndf*2) x 250
            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 5, bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 128
            nn.Conv1d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm1d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 64
            nn.Conv1d(opt.ndf * 8, opt.ndf * 16, 4, 2, 1, bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm1d(opt.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 32
            nn.Conv1d(opt.ndf * 16, opt.ndf * 32, 4, 2, 1, bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm1d(opt.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*32) x 16
            nn.Conv1d(opt.ndf * 32, opt.nz * 16, 4, 2, 1, bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm1d(opt.nz * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nz*16) x 8
            nn.Conv1d(opt.nz * 16, opt.nz, 8, 1, 0, bias=False),
            # state size. (nz) x 1
        )

    def forward(self, input):
        output = self.main(input)

        return output


class Decoder_CGAN(nn.Module):
    def __init__(self, opt):
        super(Decoder_CGAN, self).__init__()
        self.opt = opt
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(opt.nz, opt.ngf * 32, 8, 1, 0, bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm1d(opt.ngf * 32),
            nn.ReLU(True),
            # state size. (ngf*32) x8
            nn.ConvTranspose1d(opt.ngf * 32, opt.ngf * 16, 4, 2, 1, bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm1d(opt.ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x16
            nn.ConvTranspose1d(opt.ngf * 16, opt.ngf * 8, 4, 2, 1, bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm1d(opt.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 32
            nn.ConvTranspose1d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 64
            nn.ConvTranspose1d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm1d(opt.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 128
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 4, 2, 4, bias=False),
            nn.Dropout(self.opt.dropout_ratio),
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 250
            nn.ConvTranspose1d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 500
        )

    def forward(self, input):
        output = self.main(input)

        return output


class Generator_CGAN(nn.Module):
    def __init__(self, opt):
        super(Generator_CGAN, self).__init__()
        self.opt = opt
        self.beat_x = nn.Sequential(
            # input is (nc) x 500
            nn.Conv1d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 250
        )

        self.lead_label = nn.Sequential(
            # input is (num_leads) x 500
            nn.Conv1d(opt.num_leads, opt.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 250
        )

        self.encoder = Encoder_CGAN(opt)
        self.decoder = Decoder_CGAN(opt)
        # self.encoder = Encoder_BeatGAN(opt)
        # self.decoder = Decoder_BeatAGAN(opt)

    def forward(self, x, label):
        x = self.beat_x(x)
        label = self.lead_label(label)

        incat = torch.cat((x, label), dim=1)

        feature_z = self.encoder(incat)
        gen_x = self.decoder(feature_z)
        return gen_x, feature_z


class Discriminator_CGAN(nn.Module):

    def __init__(self, opt):
        super(Discriminator_CGAN, self).__init__()
        self.opt = opt

        self.beat_x = nn.Sequential(
            # input is (nc) x 500
            nn.Conv1d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 250
        )

        self.lead_label = nn.Sequential(
            # input is (num_leads) x 500
            nn.Conv1d(opt.num_leads, opt.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 250
        )

        self.encode = Encoder_CGAN(opt)

        self.fc1 = nn.Linear(opt.nz, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x, label):
        # for i in range(x.shape[0]):
        #     if x[i].shape[1] != 500:
        #         print(x[i].shape)
        x = self.beat_x(x)
        label = self.lead_label(label)

        incat = torch.cat((x, label), dim=1)

        feature_z = self.encode(incat)
        classifier = feature_z.view(-1, self.opt.nz)
        classifier = self.fc1(classifier)
        classifier = self.sig(classifier)

        return classifier, feature_z