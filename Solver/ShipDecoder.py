import os
import numpy as np
import scipy.interpolate as interp
from stl import mesh
import csv


class ShipDecoder:
    def __init__(self, PixName):
        self.PixName = PixName

    def Pix2hull(self, Imgfile, LOA, Dd, Bd, num_wl, num_x, wl_above, Outfilepath, Outfilename,PC_path,Csvfilename):

        PC = Imgfile
        Min_Pos = min(PC[:, 0])
        PC[:, 0] = PC[:, 0] - (Min_Pos)
        PC[:, 0] = PC[:, 0] / max(PC[:, 0])
        PC[:, 1] = PC[:, 1] / max(PC[:, 1])  # 半宽
        PC[:, 2] = PC[:, 2] / max(PC[:, 2])
        PC[:, 0] = PC[:, 0] * LOA
        PC[:, 1] = PC[:, 1] * Bd
        PC[:, 2] = PC[:, 2] * Dd

        f = open(PC_path + Csvfilename, 'w')
        writer = csv.writer(f, lineterminator='\n')
        for k in range(0, len(PC)):
            writer.writerow([round(x, 5) for x in PC[k]])
        f.close()
        VolumeinthisWl = np.zeros(num_wl)
        for i in range(num_wl):
            PointinthisWL = PC[i * num_x:(i + 1) * num_x, 0:]
            VolumeinthisWl[i] = np.trapz(PointinthisWL[:, 1], PointinthisWL[:, 0])
        Volume=PC[num_x+1,2]*(sum(VolumeinthisWl)-(VolumeinthisWl[0]+VolumeinthisWl[-1])/2)*2

        pts = [PC[i * num_x:(i + 1) * num_x] for i in range(num_wl)]
        x_ship_pos = np.linspace(0, max(PC[:, 0]), num_x)
        # # build deckline
        if pts[-1][0,0]>pts[num_wl-2][0,0]:
            sca=0.9
        else:
            sca=1.1
        new_data = pts[-1].copy()
        new_data[:, 2] = 1.5*Dd
        center = np.mean(new_data, axis=0)
        center[1]=0
        new_data = (new_data - center) * sca + center
        pts.append(new_data)
        NUM_WL = num_wl+1
        NUM_WL = num_wl
        for i in range(0, NUM_WL):
            indices = np.where((x_ship_pos > pts[i][0, 0]) & (x_ship_pos < pts[i][-1, 0]))[0]
            _, idx = np.unique(pts[i][:, 0], return_index=True)
            WL_curve = interp.interp1d(pts[i][idx, 0], pts[i][idx, 1], kind='linear')
            ydata = WL_curve(x_ship_pos[indices])
            pts[i] = np.vstack(
                [pts[i][0, :], np.stack([x_ship_pos[indices], ydata, pts[i][0, 2] * np.ones(len(indices))], axis=1),
                 pts[i][-1, :]])
            # start to assemble the triangles into vectors of indices from pts
            TriVec = []

        for i in range(0, NUM_WL - 1):

            # Find idx where the mesh grids begin to align between two rows returns a zero or 1:

            bow = np.argmax([pts[i][0, 0], pts[i + 1][0, 0]])

            stern = np.argmin([pts[i][-1, 0], pts[i + 1][-1, 0]])

            # Find index where mesh grid lines up and ends between each WL

            if bow:
                idx_WLB1 = 1
                idx_WLB0 = np.where(pts[i][:, 0] == pts[i + 1][idx_WLB1, 0])[0][0]
            else:
                idx_WLB0 = 1
                aaa = pts[i + 1][:, 0]
                bbb = pts[i][idx_WLB0, 0]
                idx_WLB1 = np.where(pts[i + 1][:, 0] == pts[i][idx_WLB0, 0])[0][0]

            if stern:
                idx_WLS1 = len(pts[i + 1]) - 2
                idx_WLS0 = np.where(pts[i][:, 0] == pts[i + 1][idx_WLS1, 0])[0][0]
            else:
                idx_WLS0 = len(pts[i]) - 2
                idx_WLS1 = np.where(pts[i + 1][:, 0] == pts[i][idx_WLS0, 0])[0][0]

                # check that these two are the same size:

                # Build the bow triangles Includes Port assignments

            if bow:
                TriVec.append([pts[i + 1][idx_WLB1], pts[i][0], pts[i + 1][0]])

                for j in range(0, idx_WLB0):
                    TriVec.append([pts[i + 1][idx_WLB1], pts[i][j + 1], pts[i][j]])



            else:

                for j in range(0, idx_WLB1):
                    TriVec.append([pts[i][0], pts[i + 1][j], pts[i + 1][j + 1]])

                TriVec.append([pts[i][0], pts[i + 1][idx_WLB1], pts[i][idx_WLB0]])

                # Build main part of hull triangles. Port Assignments
            for j in range(0, idx_WLS1 - idx_WLB1):
                TriVec.append([pts[i][idx_WLB0 + j], pts[i + 1][idx_WLB1 + j], pts[i + 1][idx_WLB1 + j + 1]])
                TriVec.append([pts[i][idx_WLB0 + j], pts[i + 1][idx_WLB1 + j + 1], pts[i][idx_WLB0 + j + 1]])

                # Build the stern:
            if stern:

                for j in range(idx_WLS0, len(pts[i]) - 1):
                    TriVec.append([pts[i + 1][idx_WLS1], pts[i][j + 1], pts[i][j]])

                TriVec.append([pts[i + 1][idx_WLS1], pts[i + 1][-1], pts[i][-1]])

            else:

                TriVec.append([pts[i][idx_WLS0], pts[i + 1][idx_WLS1], pts[i][-1]])

                for j in range(idx_WLS1, len(pts[i + 1]) - 1):
                    TriVec.append([pts[i][-1], pts[i + 1][j], pts[i + 1][j + 1]])

        TriVec = np.array(TriVec)

        hullTriangles = 2 * len(TriVec)
        numTriangles = hullTriangles

        z_idx = NUM_WL - wl_above - 1

        transomTriangles = 2 * wl_above - 1

        numTriangles += transomTriangles

        numTriangles += 2 * len(pts[-1]) - 3

        HULL = mesh.Mesh(np.zeros(numTriangles, dtype=mesh.Mesh.dtype))

        HULL.vectors[0:len(TriVec)] = np.copy(TriVec)


        TriVec_stbd = np.copy(TriVec[:, ::-1])
        TriVec_stbd[:, :, 1] *= -1
        HULL.vectors[len(TriVec):hullTriangles] = np.copy(TriVec_stbd)

        # NowBuild the transom:
        pts_trans = np.zeros((wl_above + 1, 3))

        for i in range(0, len(pts_trans)):
            pts_trans[i] = pts[z_idx + i][-1, :]

        pts_tranp = np.array(pts_trans)

        pts_tranp[:, 1] *= -1.0

        HULL.vectors[hullTriangles] = np.array([pts_trans[0], pts_trans[1], pts_tranp[1]])
        for i in range(1, wl_above):
            HULL.vectors[hullTriangles + 2 * i - 1] = np.array([pts_trans[i], pts_trans[i + 1], pts_tranp[i]])
            HULL.vectors[hullTriangles + 2 * i] = np.array([pts_tranp[i], pts_trans[i + 1], pts_tranp[i + 1]])

        # Add the deck lid
        pts_Lids = pts[NUM_WL - 1]

        pts_Lidp = np.array(pts_Lids)
        pts_Lidp[:, 1] *= -1.0

        startTriangles = hullTriangles + transomTriangles

        # Points are orered so the right hand rule points the lid in positive z
        HULL.vectors[startTriangles] = np.array([pts_Lids[0], pts_Lidp[1], pts_Lids[1]])

        for i in range(1, len(pts_Lids) - 1):
            HULL.vectors[startTriangles + 2 * i - 1] = np.array([pts_Lids[i], pts_Lidp[i], pts_Lids[i + 1]])
            HULL.vectors[startTriangles + 2 * i] = np.array([pts_Lids[i + 1], pts_Lidp[i], pts_Lidp[i + 1]])

        # utfilepath+Outfilename"
        os.makedirs(Outfilepath, exist_ok=True)
        HULL.save(Outfilepath + Outfilename)
        return Volume


