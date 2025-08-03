# coding:utf-8
import numpy as np
import time
from Solver.RBFN import RBFN
from Solver.Latin import latin
from Solver.GA import GA
from ShipInfoGAN.ShipinfoGAN import Generator
from Solver.ShipDecoder import ShipDecoder
import Solver.CommonMethod as CommonMethod
from Solver.ShipDecoderforcal import ShipDecoderforcal
import argparse
import torch
import os


def updatemodel(data):
    global numx0, numy0, numx1, numy1, numx2, numy2
    pre0 = model[0].predict(data)
    pre1 = model[1].predict(data)
    pre2 = model[2].predict(data)

    error = abs(pre1-pre2)
    seq = np.ravel(np.where(error == np.min(error)))[0]
    xtemp = np.row_stack((numx0, data[seq]))
    ytemp = np.append(numy0, (pre1[seq]+pre2[seq])/2)
    model[0].fit(xtemp, ytemp)
    # print('model0update')
    error = abs(pre0-pre2)
    seq = np.ravel(np.where(error == np.min(error)))[0]
    xtemp = np.row_stack((numx1, data[seq]))
    ytemp = np.append(numy1, (pre0[seq]+pre2[seq])/2)
    model[1].fit(xtemp, ytemp)
    # print('model1update')
    error = abs(pre0-pre1)
    seq = np.ravel(np.where(error == np.min(error)))[0]
    xtemp = np.row_stack((numx2, data[seq]))
    ytemp = np.append(numy2, (pre0[seq]+pre1[seq])/2)
    model[2].fit(xtemp, ytemp)
    # print('model2update')

def resetmodel(x,y):
    global numx0, numx1, numx2, numy0, numy1, numy2
    shuffledata = np.column_stack((y, x))
    np.random.shuffle(shuffledata)
    newx = shuffledata[:, 1:]
    newy = shuffledata[:, :1]
    numx0 = newx[:traindata, ]
    numy0 = newy[:traindata, ]
    numx1 = newx[traindata:2 * traindata, ]
    numy1 = newy[traindata:2 * traindata, ]
    numx2 = newx[datanum - traindata:, ]
    numy2 = newy[datanum - traindata:, ]

    model[0].fit(numx0, numy0)
    model[1].fit(numx1, numy1)
    model[2].fit(numx2, numy2)

if __name__ == '__main__':
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
    Lwl = 106.00
    Bd = 17.60 / 2
    Dd = 5.20


    dimension = 20
    lower_bound = -1
    upper_bound = 1
    data = np.loadtxt('././Sample1data.csv', delimiter=',')
    datanum = 1000
    x=data[:, :20]
    y=data[:, 20]
    model = [0] * 3
    traindata = int(datanum / 3)
    model[0] = RBFN(input_shape=dimension, hidden_shape=int(np.sqrt(traindata)), kernel='gaussian')
    model[1] = RBFN(input_shape=dimension, hidden_shape=int(np.sqrt(traindata)), kernel='gaussian')
    model[2] = RBFN(input_shape=dimension, hidden_shape=int(np.sqrt(traindata)), kernel='gaussian')
    resetmodel(x,y)

    max_iter = 50
    ga = GA(pop_size=100, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound)
    ga.init_Population()
    for i in range(max_iter):
        updatemodel(ga.pop)
        ga.crossover(ga.pc)  
        ga.mutation(ga.pm) 
        ga.pop = np.unique(ga.pop, axis=0)
        for j in range(0, 3):
            temp = model[j].predict(ga.pop)
            if j == 0:
                fit_value = temp
            else:
                fit_value = fit_value + temp
        fit_value = fit_value.reshape((len(ga.pop), 1))
        for myi in range(np.shape(ga.pop)[0]):
            z = torch.tensor(ga.pop[myi]).unsqueeze(0).float().to(device)
            label_input = CommonMethod.sampleto_categorical(np.ones(1, dtype=int), opt.n_classes).to(device)
            code_input = 0.742976069 * torch.ones(1, opt.code_dim, device=device)
            ships = ship_generator(z, label_input, code_input)
            X = ships[0, 0, :, :].detach().cpu().numpy()
            Y = ships[0, 1, :, :].detach().cpu().numpy()
            Z = np.linspace(0, 1, 40).round(3)  # 40 if num_wl
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
            myshipdecoder = ShipDecoderforcal("D")

            shipDis = myshipdecoder.Pix2hull(mypts, Lwl, Dd, Bd, 40, 40)
            input_displacement = code_input * Lwl * Bd * 2 * Dd
            input_displacement = input_displacement.cpu().tolist()
            DisplacementError = np.abs(shipDis - np.ravel(input_displacement)) / np.ravel(input_displacement)
            if np.abs(DisplacementError) > 0.01:
                fit_value[myi]=fit_value[myi]+1000000

        ga.selection(fit_value)  
        resetmodel(x,y)
        print('minfitness:', np.min(fit_value))

    optimum = ga.first[-1]
    print('Optimal solution :', optimum)
    z = torch.tensor(optimum).unsqueeze(0).float().to(device)
    label_input = CommonMethod.sampleto_categorical(np.ones(1, dtype=int), opt.n_classes).to(device)
    code_input = 0.742976069 * torch.ones(1, opt.code_dim, device=device)
    ships = ship_generator(z, label_input, code_input)
    X = ships[0, 0, :, :].detach().cpu().numpy()
    Y = ships[0, 1, :, :].detach().cpu().numpy()
    Z = np.linspace(0, 1, 40).round(3)  # 40 if num_wl
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
    myshipdecoder = ShipDecoderforcal("D")
    myshipdecoder = ShipDecoder("D")
    # build model
    Geometry_path = ".//Geometry1//"
    if not os.path.exists(Geometry_path):
        os.mkdir(Geometry_path)
    PC_path = ".//PC1//"
    if not os.path.exists(PC_path):
        os.mkdir(PC_path)
    shipDis = myshipdecoder.Pix2hull(mypts, Lwl, Dd, Bd, 40, 40, 40 - wfb[0], Geometry_path,
                                     "Optship.stl", "Opt.stl", PC_path,
                                     "Optship.csv")
    print('Optimal solution Dis :', shipDis)


