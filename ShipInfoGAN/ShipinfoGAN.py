import torch.nn as nn
import torch
from torch.nn.utils import spectral_norm


class FiLM(nn.Module):
    def __init__(self, num_features, code_dim):
        super().__init__()
        self.gamma = nn.Linear(code_dim, num_features)
        self.beta = nn.Linear(code_dim, num_features)

    def forward(self, x, code):
        gamma = self.gamma(code).unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        beta = self.beta(code).unsqueeze(2).unsqueeze(3)    # (B, C, 1, 1)
        return x * gamma + beta

class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim
        self.init_size = 5
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128 * self.init_size ** 2),
            nn.LayerNorm(128 * self.init_size ** 2),
            nn.SiLU(inplace=True)
        )
        self.init_reshape = lambda x: x.view(x.size(0), 128, self.init_size, self.init_size)
        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(32, opt.channels, 3, padding=1),
            nn.Sigmoid()
        )
        self.film1 = FiLM(128, opt.code_dim)
        self.film2 = FiLM(64, opt.code_dim)
        self.film3 = FiLM(32, opt.code_dim)

    def forward(self, z, labels, code):
        x = torch.cat((z, labels, code), dim=1)
        x = self.fc(x)
        x = self.init_reshape(x)
        x = self.block1(x)
        x = self.film1(x, code)
        x = self.block2(x)
        x = self.film2(x, code)
        x = self.block3(x)
        x = self.film3(x, code)
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        def block(in_c, out_c):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, out_c, 3, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.3)
            )

        self.conv = nn.Sequential(
            block(opt.channels, 32),
            block(32, 64),
            block(64, 128)
        )

        self.flatten = nn.Flatten()
        self.adv_layer = nn.Linear(128 * 5 * 5, 1)

        self.q_shared = nn.Sequential(
            nn.Linear(128 * 5 * 5, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        self.q_cat = nn.Linear(128, opt.n_classes)
        self.q_cont = nn.Linear(128, opt.code_dim)  #

    def forward(self, img):
        x = self.conv(img)
        x = self.flatten(x)
        validity = self.adv_layer(x)
        q = self.q_shared(x)
        pred_label = self.q_cat(q)
        pred_code = self.q_cont(q)
        return validity, pred_label, pred_code
