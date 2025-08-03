from ShipInfoGAN.ShipinfoGAN import Generator
from Solver.ShipDecoder import ShipDecoder
import Solver.CommonMethod as CommonMethod
import argparse
import torch
import os
import csv
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=501)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--n_cpu", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=20)
parser.add_argument("--code_dim", type=int, default=1)
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--img_size", type=int, default=40)
parser.add_argument("--channels", type=int, default=2)
parser.add_argument("--sample_interval", type=int, default=400)
opt = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ship_generator = Generator(opt).to(device)

tainingmodel = torch.load("ShipinfoGAN20D.pth", map_location=device)
ship_generator.load_state_dict(tainingmodel['generator'])
ship_generator.eval()

ship_num=6
z = torch.randn(ship_num, opt.latent_dim, device=device)
label_input = CommonMethod.sampleto_categorical(np.ones(ship_num,dtype=int), opt.n_classes).to(device)
code_input = 0.8 * torch.ones(ship_num, opt.code_dim, device=device)
# Seting ship main dimensions
Lwl=90
Bd=15.6/2
Dd=5.6
ships = ship_generator(z, label_input, code_input)

# cal displacement
input_displacement=code_input.detach().cpu().numpy() *Lwl*Bd*2*Dd


#build model
Geometry_path=".//Geometry_Phy_Bulbbow//"
if not os.path.exists(Geometry_path):
    os.mkdir(Geometry_path)
PC_path = ".//PC_Phy_Bulbbow//"
if not os.path.exists(PC_path):
    os.mkdir(PC_path)
Displacement=[]

for i in range(ship_num):
    X = ships[i, 0, :, :].detach().cpu().numpy()
    Y = ships[i, 1, :, :].detach().cpu().numpy()
    Z = np.linspace(0, 1, 40).round(3) #40 if num_wl
    Z_matrix = np.tile(Z.reshape(-1, 1), (1, 40))  # 40行40列
    mypts = np.empty((0, 3))
    wfb = []
    for j in range(40):
        x_line = X[j, :]
        y_line = Y[j, :]
        if y_line[-1] >= 1e-2:
            wfb.append(j)
        z_line = np.full_like(x_line, Z[j])
        new_pts = np.stack([x_line, y_line, z_line], axis=1)
        if mypts is None or mypts.size == 0:
            mypts = new_pts
        else:
            mypts = np.vstack((mypts, new_pts))
    wfb.append(40)

    myshipdecoder = ShipDecoder("20D")
    shipDis=myshipdecoder.Pix2hull(mypts, Lwl, Dd, Bd, 40, 40, 40 - wfb[0], Geometry_path,"SampleHullD"+str(i)+".stl",PC_path,"SampleHullD"+str(i)+".csv")
    Displacement.append(shipDis)

#Post-processing
DisplacementError = (np.ravel(Displacement) - np.ravel(input_displacement)) / np.ravel(input_displacement)
combined = zip(Displacement,input_displacement,DisplacementError)
# to file
result_path=".//result_Phy_Bulbbow//"
if not os.path.exists(result_path):
    os.mkdir(result_path)
with open(result_path+"output.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Displacement','input_displacement','DisplacementError'])
    writer.writerows(combined)



