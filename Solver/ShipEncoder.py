import os
import numpy as np
import csv

"""
This part is used to build training data
"""
class ShipEncoder:
    def __init__(self, ShipName):
        self.ShipName=ShipName

    def Hull2Pix(self, PC, DataPath, filename,num_wl,num_x):
        # PC is the point cloud of ship
        PC = np.round(PC, 5)
        Min_Pos = min(PC[:, 0])
        PC[:, 0] = PC[:, 0] - (Min_Pos)
        VolumeinthisWl = np.zeros(num_wl)
        for i in range(num_wl):
            PointinthisWL = PC[i * num_x:(i + 1) * num_x, 0:]
            VolumeinthisWl[i] = np.trapz(PointinthisWL[:, 1], PointinthisWL[:, 0])
        Volume=PC[num_x+1,2]*(sum(VolumeinthisWl)-(VolumeinthisWl[0]+VolumeinthisWl[-1])/2)*2
        Cb=Volume/max(PC[:, 1])/max(PC[:, 2])/max(PC[:, 0])/2
        PC[:, 0] = PC[:, 0]/max(PC[:, 0])
        PC[:, 1] = PC[:, 1]/max(PC[:, 1]) #半宽
        PC[:, 2] = PC[:, 2]/max(PC[:, 2])
        os.makedirs(DataPath, exist_ok=True)
        f = open(DataPath + filename + "_Cb={:.5f}.csv".format(Cb), 'w')
        writer = csv.writer(f, lineterminator='\n')
        for k in range(0, len(PC)):
            writer.writerow([round(x, 5) for x in PC[k]])
        f.close()
        return Cb



