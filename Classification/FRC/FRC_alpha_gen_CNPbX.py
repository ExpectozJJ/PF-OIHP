import numpy as np
import math
import scipy.io as sio
from GeneralisedFormanRicci.frc import GeneralisedFormanRicci, n_faces
import time
import sys
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import scipy
import os
import umap
import umap.plot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#folder = sys.argv[1]

x = ["Br", "Cl", "I"]
sym = ["Cubic", "Orthorhombic", "Tetragonal"]

labels = []
for (a,b) in itertools.product(x,sym):
    labels.append(a+"-"+b)
    #print("MAPb{}3_{}_CNXPb_data".format(a, b))
"""
begin = 501; last = 1001
ndata = 1; fmax = 4.5; step = 0.1

os.chdir("/mnt/c/Users/wee_j/Documents/PhD Research/PhD Codes/Vijai_Perovskite_2021/HOIP_Classification/HOIP_Classification")
test_results = []
#for (a,b) in itertools.product(x, sym):
    #folder = "MAPb{}3_{}_CNXPb_data".format(a, b)
tmp = []
for idx in range(begin, last):
    print(idx)
    statdata = [[ 0 for i in range(1) ] for j in range(ndata) ]
    for ii in range(ndata):
        fno=ii+1
        # fpath = './AABBO_ESdata_1/'
        # fname = fpath + str(fno) + '/' + str(fno) + '_A1.txt'
        # file = open(fname)
        file = open('./'+folder+'/CNXPb_atmlist_L5_f{}.txt'.format(idx))

        contents = file.readlines()
        for i in range(len(contents)):
            contents[i] = contents[i].rstrip("\n").split(",")
            contents[i] = [float(s) for s in contents[i]]

        #print(contents)
        out = []
        for f in np.arange(fmax, fmax+0.1, step):
            sc = GeneralisedFormanRicci(points=contents, epsilon=f, method="alpha", p=2)
            frc = sc.compute_forman()
            tmp.append([list(frc[0].values()), list(frc[1].values()), list(frc[2].values())])
test_results.append(tmp)

np.save("./"+folder+"_feat.npy", test_results)
"""

data = []
for (a,b) in itertools.product(x,sym):
    folder = "MAPb{}3_{}_CNXPb_data".format(a, b)
    data.append(np.load('./'+folder+'_feat.npy', allow_pickle=True)[0])

median_pts = []
for i in range(9):
    pts = []
    for k in range(3):
        tmp = []
        plt.figure(figsize=(10, 10))
        for j in range(500):
            ax = sns.kdeplot(data[i][j][k], bw_method=.15, gridsize=200, label = labels[i])
            x = list(ax.lines[j].get_data()[0])
            y = list(ax.lines[j].get_data()[1])
            cdf = scipy.integrate.cumtrapz(y, x, initial=0)
            nearest_05 = np.abs(cdf-0.5).argmin()

            x_median = x[nearest_05]
            y_median = y[nearest_05]
            tmp.append([x_median, y_median])
        plt.close()
        pts.append(tmp)
    median_pts.append(pts)

print(np.shape(median_pts))
frc_all = np.reshape(np.transpose(median_pts, axes=(0,2,1,3)), (4500, 6))

frd = 1000; frs = 500
pca = PCA(n_components=2)
values = pca.fit_transform(frc_all)

plt.figure(dpi=200)
plt.scatter(-values[:frd-frs+1,0], values[:frd-frs+1,1], marker='^', color='white', alpha=0.75, edgecolor='tab:blue', linewidth=.5, s=20, label="Br-Cubic")
plt.scatter(-values[frd-frs+1:2*(frd-frs+1),0], values[frd-frs+1:2*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:orange', linewidth=0.5, s=20, label="Br-Ortho")
plt.scatter(-values[2*(frd-frs+1):3*(frd-frs+1),0], values[2*(frd-frs+1):3*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:green', linewidth=0.5, s=20, label="Br-Tetra")

plt.scatter(-values[3*(frd-frs+1):4*(frd-frs+1),0], values[3*(frd-frs+1):4*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:red', linewidth=0.5, s=20, label="Cl-Cubic")
plt.scatter(-values[4*(frd-frs+1):5*(frd-frs+1),0], values[4*(frd-frs+1):5*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:purple', linewidth=0.5, s=20, label="Cl-Ortho")
plt.scatter(-values[5*(frd-frs+1):6*(frd-frs+1),0], values[5*(frd-frs+1):6*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:brown', linewidth=0.5, s=20, label="Cl-Tetra")

plt.scatter(-values[6*(frd-frs+1):7*(frd-frs+1),0], values[6*(frd-frs+1):7*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:pink', linewidth=0.5, s=20, label="I-Cubic")
plt.scatter(-values[7*(frd-frs+1):8*(frd-frs+1),0], values[7*(frd-frs+1):8*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:gray', linewidth=0.5, s=20, label="I-Ortho")
plt.scatter(-values[8*(frd-frs+1):9*(frd-frs+1),0], values[8*(frd-frs+1):9*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:olive', linewidth=0.5, s=20, label="I-Tetra")

#plt.ylim(-400, 700)
#plt.xlim(-1000, 1000)
plt.legend(ncol=3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
plt.savefig("MAPbX3_Classification_CNPbX_PCA.png", dpi=200)
plt.show()

values = umap.UMAP(min_dist=1, metric='euclidean').fit_transform(frc_all)
plt.figure(dpi=200)
plt.scatter(values[:frd-frs+1,0], values[:frd-frs+1,1], marker='^', color='white', alpha=0.75, edgecolor='tab:blue', linewidth=.5, s=20, label="Br-Cubic")
plt.scatter(values[frd-frs+1:2*(frd-frs+1),0], values[frd-frs+1:2*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:orange', linewidth=0.5, s=20, label="Br-Ortho")
plt.scatter(values[2*(frd-frs+1):3*(frd-frs+1),0], values[2*(frd-frs+1):3*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:green', linewidth=0.5, s=20, label="Br-Tetra")

plt.scatter(values[3*(frd-frs+1):4*(frd-frs+1),0], values[3*(frd-frs+1):4*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:red', linewidth=0.5, s=20, label="Cl-Cubic")
plt.scatter(values[4*(frd-frs+1):5*(frd-frs+1),0], values[4*(frd-frs+1):5*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:purple', linewidth=0.5, s=20, label="Cl-Ortho")
plt.scatter(values[5*(frd-frs+1):6*(frd-frs+1),0], values[5*(frd-frs+1):6*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:brown', linewidth=0.5, s=20, label="Cl-Tetra")

plt.scatter(values[6*(frd-frs+1):7*(frd-frs+1),0], values[6*(frd-frs+1):7*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:pink', linewidth=0.5, s=20, label="I-Cubic")
plt.scatter(values[7*(frd-frs+1):8*(frd-frs+1),0], values[7*(frd-frs+1):8*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:gray', linewidth=0.5, s=20, label="I-Ortho")
plt.scatter(values[8*(frd-frs+1):9*(frd-frs+1),0], values[8*(frd-frs+1):9*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:olive', linewidth=0.5, s=20, label="I-Tetra")

#plt.ylim(-10, 35)
#plt.xlim(-80, 80)
plt.legend(ncol=3, loc='upper left', handlelength=.25, borderpad=.25, fontsize=10)
plt.savefig("MAPbX3_Classification_CNPbX_UMAP.png", dpi=200)
plt.show()



values = TSNE(n_components=2).fit_transform(frc_all)
plt.figure(dpi=200)
plt.scatter(values[:frd-frs+1,0], values[:frd-frs+1,1], marker='^', color='white', alpha=0.75, edgecolor='tab:blue', linewidth=.5, s=20, label="Br-Cubic")
plt.scatter(values[frd-frs+1:2*(frd-frs+1),0], values[frd-frs+1:2*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:orange', linewidth=0.5, s=20, label="Br-Ortho")
plt.scatter(values[2*(frd-frs+1):3*(frd-frs+1),0], values[2*(frd-frs+1):3*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:green', linewidth=0.5, s=20, label="Br-Tetra")

plt.scatter(values[3*(frd-frs+1):4*(frd-frs+1),0], values[3*(frd-frs+1):4*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:red', linewidth=0.5, s=20, label="Cl-Cubic")
plt.scatter(values[4*(frd-frs+1):5*(frd-frs+1),0], values[4*(frd-frs+1):5*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:purple', linewidth=0.5, s=20, label="Cl-Ortho")
plt.scatter(values[5*(frd-frs+1):6*(frd-frs+1),0], values[5*(frd-frs+1):6*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:brown', linewidth=0.5, s=20, label="Cl-Tetra")

plt.scatter(values[6*(frd-frs+1):7*(frd-frs+1),0], values[6*(frd-frs+1):7*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:pink', linewidth=0.5, s=20, label="I-Cubic")
plt.scatter(values[7*(frd-frs+1):8*(frd-frs+1),0], values[7*(frd-frs+1):8*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:gray', linewidth=0.5, s=20, label="I-Ortho")
plt.scatter(values[8*(frd-frs+1):9*(frd-frs+1),0], values[8*(frd-frs+1):9*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:olive', linewidth=0.5, s=20, label="I-Tetra")

#plt.ylim(-80, 140)
#plt.xlim(-80, 80)
plt.legend(ncol=3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
plt.savefig("MAPbX3_Classification_CNPbX_tSNE.png", dpi=200)
plt.show()



    
        
