import torch
import os
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def sampleto_categorical(y, num_columns):
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0
    return torch.tensor(y_cat, dtype=torch.float32)

def curvature_line(x, y):
    t = np.arange(len(x))


    fx = interp1d(t, x, kind='linear')
    fy = interp1d(t, y, kind='linear')


    t_fine = np.linspace(t[0], t[-1], 40)
    x_fine = fx(t_fine)
    y_fine = fy(t_fine)


    dx = np.gradient(x_fine, t_fine)
    dy = np.gradient(y_fine, t_fine)

    ddx = np.gradient(dx, t_fine)
    ddy = np.gradient(dy, t_fine)


    kappa = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

    return t_fine, kappa

def show_WL_curvaturerate(data,WL_curvatureratepath,ship_num):
    data=data.round(3)
    all_rows = []
    unique_z = np.unique(data[:, 2])
    for i, z_val in enumerate(unique_z):
        layer = data[data[:, 2] == z_val]
        x, y = layer[:, 0], layer[:, 1]


        t_fine, kappa = curvature_line(x, y)


        dx = np.gradient(x)
        dy = np.gradient(y)
        ds = np.sqrt(dx ** 2 + dy ** 2)
        s = np.cumsum(ds)
        s = np.interp(t_fine, np.arange(len(x)), s)


        dkappa_ds = np.abs(np.gradient(kappa, s))
        all_rows.append(dkappa_ds)
    all_rows = np.vstack(all_rows)
    max_rate=np.max(all_rows)
    # 保存为一个 CSV 文件
    df = pd.DataFrame(all_rows)

    if not os.path.exists(WL_curvatureratepath):
        os.mkdir(WL_curvatureratepath)
    df.to_csv(WL_curvatureratepath+"Ship"+str(ship_num)+"result.csv")

    return max_rate


def curvature_surf(grid_x, grid_y,grid_z,Surf_curvatureratepath,ship_num):
    dz_dx, dz_dy = np.gradient(grid_z, grid_x[0, :], grid_y[:, 0])
