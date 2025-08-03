import numpy as np


class ShipDecoderforcal:
    def __init__(self, PixName):
        self.PixName = PixName

    def Pix2hull(self, Imgfile, LOA, Dd, Bd, num_wl, num_x):

        PC = Imgfile
        Min_Pos = min(PC[:, 0])
        PC[:, 0] = PC[:, 0] - (Min_Pos)
        PC[:, 0] = PC[:, 0] / max(PC[:, 0])
        PC[:, 1] = PC[:, 1] / max(PC[:, 1])  # 半宽
        PC[:, 2] = PC[:, 2] / max(PC[:, 2])
        PC[:, 0] = PC[:, 0] * LOA
        PC[:, 1] = PC[:, 1] * Bd
        PC[:, 2] = PC[:, 2] * Dd

        VolumeinthisWl = np.zeros(num_wl)
        for i in range(num_wl):
            PointinthisWL = PC[i * num_x:(i + 1) * num_x, 0:]
            VolumeinthisWl[i] = np.trapz(PointinthisWL[:, 1], PointinthisWL[:, 0])
        Volume=PC[num_x+1,2]*(sum(VolumeinthisWl)-(VolumeinthisWl[0]+VolumeinthisWl[-1])/2)*2

        return Volume


