import numpy as np
import torch
import itertools
from HullDataloder import CsvHullDataset
from ShipinfoGAN import Generator, Discriminator
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt



def sampleto_categorical_torch(y, num_columns):
    y_cat = torch.zeros((y.shape[0], num_columns), device=y.device)
    y_cat.scatter_(1, y.unsqueeze(1), 1.0)
    return y_cat
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
def smoothness_loss(img):
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return (dx**2).mean() + (dy**2).mean()
def show_images(images, a, b):
    for i in range(1):
        X = images[i, 0, :, :].detach().cpu()
        Y = images[i, 1, :, :].detach().cpu()
        Z = np.linspace(0, 1, 40).round(3)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for j in range(40):
            x_line = X[j, :].numpy()
            y_line = Y[j, :].numpy()
            z_line = np.full_like(x_line, Z[j])
            ax.plot(x_line, y_line, z_line, color='b', linewidth=1.0, alpha=0.7)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Waterlines - Sample {i + 1}')
        plt.tight_layout()
        plt.savefig(f".//Mypix_40WL_imp5//my_plotepoch{a}iter={b}img{i}.png")
        plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=301)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--n_cpu", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=5)
parser.add_argument("--code_dim", type=int, default=1)
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--img_size", type=int, default=40)
parser.add_argument("--channels", type=int, default=2)
parser.add_argument("--sample_interval", type=int, default=400)
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
root = './/Geometry_40WL//Image//'
dataset = CsvHullDataset(root_dir=root)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

# Models
ship_generator = Generator(opt).to(device)
discriminator = Discriminator(opt).to(device)

adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)
# adversarial_loss = torch.nn.MSELoss().to(device)
categorical_loss = torch.nn.CrossEntropyLoss().to(device)
continuous_loss = torch.nn.SmoothL1Loss().to(device)

ship_generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(ship_generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(),  lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(itertools.chain(ship_generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))


start_epoch = 0

for epoch in range(start_epoch, opt.n_epochs):
    for i, (ship_image_tensor, ship_image_label, ship_Cb) in enumerate(dataloader):
        ship_image_tensor = ship_image_tensor.to(device)
        ship_image_label = ship_image_label.to(device)
        ship_Cb = ship_Cb.to(device)

        valid = torch.ones(opt.batch_size, 1, device=device) * 0.95  # from 1 → 0.9
        fake = torch.zeros(opt.batch_size, 1, device=device) +0.05 # from 0 → 0.1

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(opt.batch_size, opt.latent_dim, device=device)

        sampled_labels = torch.randint(0, opt.n_classes, (opt.batch_size,), device=device)
        label_input = sampleto_categorical_torch(sampled_labels, opt.n_classes)
        code_input = 0.25 +  torch.rand(opt.batch_size, opt.code_dim, device=device)* (0.99 - 0.25)
        gen_imgs = ship_generator(z, label_input, code_input)
        validity, _, _ = discriminator(gen_imgs)

        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # Train Discriminat  or
        if i%1==0:
            optimizer_D.zero_grad()
            real_pred, _, _ = discriminator(ship_image_tensor)
            d_real_loss = adversarial_loss(real_pred, valid)
            fake_pred, _, _ = discriminator(gen_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        # Information Loss
        optimizer_info.zero_grad()
        sampled_labels = torch.randint(0, opt.n_classes, (opt.batch_size,), device=device)
        gt_labels = sampled_labels.long()
        z = torch.randn(opt.batch_size, opt.latent_dim, device=device)

        label_input = sampleto_categorical_torch(sampled_labels, opt.n_classes)
        code_input = 0.25 +  torch.rand(opt.batch_size, opt.code_dim, device=device)* (0.99 - 0.25)
        gen_imgs = ship_generator(z, label_input, code_input)
        _, pred_label, pred_code = discriminator(gen_imgs)
        g_lossadd = continuous_loss(pred_code, code_input)
        info_loss = 0.2*categorical_loss(pred_label, gt_labels)+0.8*g_lossadd
        info_loss.backward()
        optimizer_info.step()

        print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [info loss: {info_loss.item():.6f}] [G lossadd: {g_lossadd.item():.6f}]")
        f.write(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [info loss: {info_loss.item():.6f}]\n")
