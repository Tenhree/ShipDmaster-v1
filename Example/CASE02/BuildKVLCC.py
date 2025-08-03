from ShipInfoGAN.ShipinfoGAN import Generator
from Solver.ShipDecoder import ShipDecoder
import Solver.CommonMethod as CommonMethod
import argparse
import torch
import os
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 使 cudnn 可复现
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
data = np.loadtxt("KVLCChull.csv", delimiter=',')
# Seting ship main dimensions
Min_Pos = min(data[:, 0])
data[:, 0] = data[:, 0] - (Min_Pos)
Lwl = np.max(data[:, 0])
Bd = np.max(data[:, 1])
Dd = np.max(data[:, 2])




ship_num=1
#build model
Geometry_path=".//Geometry//"
if not os.path.exists(Geometry_path):
    os.mkdir(Geometry_path)
PC_path = ".//PC//"
if not os.path.exists(PC_path):
    os.mkdir(PC_path)


x = [-2.93184509, -7.73303597,  7.08077966, -1.81000543,  7.44955524 , 4.53407025,
 -3.55596674, -8.62538225 , 3.93928406, -2.63583181, -0.75826954, -9.01649356,
  2.647019 ,  -0.47703628, -6.52732095, -2.4433769 ,  6.33973538 ,-3.45393193,
 -8.29488567  ,9.06597308  ,0.88262357]
z = torch.tensor(x[0:20]).unsqueeze(0).float().to(device)
code_input = x[20] * torch.ones(1, opt.code_dim, device=device)
label_input = CommonMethod.sampleto_categorical(np.zeros(1, dtype=int), opt.n_classes).to(device)

ships = ship_generator(z, label_input, code_input)

# cal displacement
input_displacement = code_input * Lwl * Bd * 2 * Dd
input_displacement = input_displacement.cpu().tolist()
X = ships[0, 0, :, :].detach().cpu().numpy()
Y = ships[0, 1, :, :].detach().cpu().numpy()
Z = np.linspace(0, 1, 40).round(3)
Z_matrix = np.tile(Z.reshape(-1, 1), (1, 40))
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

myshipdecoder = ShipDecoder("D")
shipDis = myshipdecoder.Pix2hull(mypts, Lwl, Dd, Bd, 40, 40, 40 - wfb[0], Geometry_path, "KVLCCfromShipInfoGAN.stl",
                                 PC_path, "KVLCCfromShipInfoGAN.csv")
