import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
from GeneralisedFormanRicci.frc import GeneralisedFormanRicci, n_faces
import os
import sys
import umap
import umap.plot
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import openTSNE
import scipy.io as sio

MAPbBr3_Cubic_CNPbXdata = []
MAPbCl3_Cubic_CNPbXdata = []
MAPbI3_Cubic_CNPbXdata = []
MAPbBr3_Ortho_CNPbXdata = []
MAPbCl3_Ortho_CNPbXdata = []
MAPbI3_Ortho_CNPbXdata = []
MAPbBr3_Tetra_CNPbXdata = []
MAPbCl3_Tetra_CNPbXdata = []
MAPbI3_Tetra_CNPbXdata = []

MAPbBr3_Cubic_CHNPbXdata = []
MAPbCl3_Cubic_CHNPbXdata = []
MAPbI3_Cubic_CHNPbXdata = []
MAPbBr3_Ortho_CHNPbXdata = []
MAPbCl3_Ortho_CHNPbXdata = []
MAPbI3_Ortho_CHNPbXdata = []
MAPbBr3_Tetra_CHNPbXdata = []
MAPbCl3_Tetra_CHNPbXdata = []
MAPbI3_Tetra_CHNPbXdata = []

print("Archiving Features...")

#### CNPbX Archive
os.chdir("./MAPbBr3_Cubic_CNXPb_data/")
os.system("tar -czvf MAPbBr3_Cubic_CNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbCl3_Cubic_CNXPb_data/")
os.system("tar -czvf MAPbCl3_Cubic_CNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbI3_Cubic_CNXPb_data/")
os.system("tar -czvf MAPbI3_Cubic_CNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbBr3_Tetragonal_CNXPb_data/")
os.system("tar -czvf MAPbBr3_Tetra_CNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbCl3_Tetragonal_CNXPb_data/")
os.system("tar -czvf MAPbCl3_Tetra_CNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbI3_Tetragonal_CNXPb_data/")
os.system("tar -czvf MAPbI3_Tetra_CNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbBr3_Orthorhombic_CNXPb_data/")
os.system("tar -czvf MAPbBr3_Ortho_CNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbCl3_Orthorhombic_CNXPb_data/")
os.system("tar -czvf MAPbCl3_Ortho_CNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbI3_Orthorhombic_CNXPb_data/")
os.system("tar -czvf MAPbI3_Ortho_CNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

#### CHNPbX Archive

os.chdir("./MAPbBr3_Cubic_CHNPbXdata/")
os.system("tar -czvf MAPbBr3_Cubic_CHNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbCl3_Cubic_CHNPbXdata/")
os.system("tar -czvf MAPbCl3_Cubic_CHNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbI3_Cubic_CHNPbXdata/")
os.system("tar -czvf MAPbI3_Cubic_CHNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbBr3_Tetragonal_CHNPbXdata/")
os.system("tar -czvf MAPbBr3_Tetra_CHNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbCl3_Tetragonal_CHNPbXdata/")
os.system("tar -czvf MAPbCl3_Tetra_CHNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbI3_Tetragonal_CHNPbXdata/")
os.system("tar -czvf MAPbI3_Tetra_CHNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbBr3_Orthorhombic_CHNPbXdata/")
os.system("tar -czvf MAPbBr3_Ortho_CHNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbCl3_Orthorhombic_CHNPbXdata/")
os.system("tar -czvf MAPbCl3_Ortho_CHNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

os.chdir("./MAPbI3_Orthorhombic_CHNPbXdata/")
os.system("tar -czvf MAPbI3_Ortho_CHNPbX_frcstat.tar.gz *.npz")
os.chdir("..")

print("Archive Completed!!")

for i in range(501, 1001):
    MAPbBr3_Cubic_CNPbXdata.append(np.load("./MAPbBr3_Cubic_CNXPb_data/CNXPb_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbCl3_Cubic_CNPbXdata.append(np.load("./MAPbCl3_Cubic_CNXPb_data/CNXPb_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbI3_Cubic_CNPbXdata.append(np.load("./MAPbI3_Cubic_CNXPb_data/CNXPb_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbBr3_Tetra_CNPbXdata.append(np.load("./MAPbBr3_Tetragonal_CNXPb_data/CNXPb_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbCl3_Tetra_CNPbXdata.append(np.load("./MAPbCl3_Tetragonal_CNXPb_data/CNXPb_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbI3_Tetra_CNPbXdata.append(np.load("./MAPbI3_Tetragonal_CNXPb_data/CNXPb_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbBr3_Ortho_CNPbXdata.append(np.load("./MAPbBr3_Orthorhombic_CNXPb_data/CNXPb_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbCl3_Ortho_CNPbXdata.append(np.load("./MAPbCl3_Orthorhombic_CNXPb_data/CNXPb_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbI3_Ortho_CNPbXdata.append(np.load("./MAPbI3_Orthorhombic_CNXPb_data/CNXPb_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])

    MAPbBr3_Cubic_CHNPbXdata.append(np.load("./MAPbBr3_Cubic_CHNPbXdata/CHNPbX_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbCl3_Cubic_CHNPbXdata.append(np.load("./MAPbCl3_Cubic_CHNPbXdata/CHNPbX_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])

    try:
        np.reshape(np.load("./MAPbCl3_Cubic_CHNPbXdata/CHNPbX_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"], (690))
    except:
        print(i)

    MAPbI3_Cubic_CHNPbXdata.append(np.load("./MAPbI3_Cubic_CHNPbXdata/CHNPbX_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbBr3_Tetra_CHNPbXdata.append(np.load("./MAPbBr3_Tetragonal_CHNPbXdata/CHNPbX_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbCl3_Tetra_CHNPbXdata.append(np.load("./MAPbCl3_Tetragonal_CHNPbXdata/CHNPbX_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbI3_Tetra_CHNPbXdata.append(np.load("./MAPbI3_Tetragonal_CHNPbXdata/CHNPbX_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbBr3_Ortho_CHNPbXdata.append(np.load("./MAPbBr3_Orthorhombic_CHNPbXdata/CHNPbX_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbCl3_Ortho_CHNPbXdata.append(np.load("./MAPbCl3_Orthorhombic_CHNPbXdata/CHNPbX_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])
    MAPbI3_Ortho_CHNPbXdata.append(np.load("./MAPbI3_Orthorhombic_CHNPbXdata/CHNPbX_atmlist_L5_f{}_frcstat.npz".format(i))["fdata"])

frs = 500
frd = 1000

ligs = ['Br-Cubic', 'Br-Ortho', 'Br-Tetra', 'Cl-Cubic', 'Cl-Ortho', 'Cl-Tetra', 'I-Cubic', 'I-Ortho', 'I-Tetra']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

for typ1 in ['Br', 'Cl', 'I']:
    for typ2 in ['Cubic', 'Ortho', 'Tetra']:
        #vars()['MAPb{}3_{}_CNPbXdata'.format(typ1, typ2)] = np.reshape(vars()['MAPb{}3_{}_CNPbXdata'.format(typ1, typ2)], (500, 11, 3, 10))
        #vars()['MAPb{}3_{}_CHNPbXdata'.format(typ1, typ2)] = np.reshape(vars()['MAPb{}3_{}_CHNPbXdata'.format(typ1, typ2)], (500, 11, 3, 10))
        #vars()['MAPb{}3_{}_CNPbXdata'.format(typ1, typ2)] = np.concatenate(((vars()['MAPb{}3_{}_CNPbXdata'.format(typ1, typ2)])[:, 2:4, 0, :], (vars()['MAPb{}3_{}_CNPbXdata'.format(typ1, typ2)])[:, 3:5, 1, :], (vars()['MAPb{}3_{}_CNPbXdata'.format(typ1, typ2)])[:, 4:6, 2, :]), axis=1)
        #vars()['MAPb{}3_{}_CHNPbXdata'.format(typ1, typ2)] = np.concatenate(((vars()['MAPb{}3_{}_CHNPbXdata'.format(typ1, typ2)])[:, 2:4, 0, :], (vars()['MAPb{}3_{}_CHNPbXdata'.format(typ1, typ2)])[:, 3:5, 1, :], (vars()['MAPb{}3_{}_CHNPbXdata'.format(typ1, typ2)])[:, 4:6, 2, :]), axis=1)
        print(typ1, typ2)
        vars()['MAPb{}3_{}_CNPbXdata'.format(typ1, typ2)] = np.reshape(vars()['MAPb{}3_{}_CNPbXdata'.format(typ1, typ2)], (500, 23*3*10))
        print(np.shape(vars()['MAPb{}3_{}_CNPbXdata'.format(typ1, typ2)]))

        vars()['MAPb{}3_{}_CHNPbXdata'.format(typ1, typ2)] = np.reshape(vars()['MAPb{}3_{}_CHNPbXdata'.format(typ1, typ2)], (500, 23*3*10))
        print(np.shape(vars()['MAPb{}3_{}_CHNPbXdata'.format(typ1, typ2)]))
        

MAPbBr3_Cubic = np.concatenate((MAPbBr3_Cubic_CNPbXdata, MAPbBr3_Cubic_CHNPbXdata), axis=1)
MAPbBr3_Ortho = np.concatenate((MAPbBr3_Ortho_CNPbXdata, MAPbBr3_Ortho_CHNPbXdata), axis=1)
MAPbBr3_Tetra = np.concatenate((MAPbBr3_Tetra_CNPbXdata, MAPbBr3_Tetra_CHNPbXdata), axis=1)

MAPbCl3_Cubic = np.concatenate((MAPbCl3_Cubic_CNPbXdata, MAPbCl3_Cubic_CHNPbXdata), axis=1)
MAPbCl3_Ortho = np.concatenate((MAPbCl3_Ortho_CNPbXdata, MAPbCl3_Ortho_CHNPbXdata), axis=1)
MAPbCl3_Tetra = np.concatenate((MAPbCl3_Tetra_CNPbXdata, MAPbCl3_Tetra_CHNPbXdata), axis=1)

MAPbI3_Cubic = np.concatenate((MAPbI3_Cubic_CNPbXdata, MAPbI3_Cubic_CHNPbXdata), axis=1)
MAPbI3_Ortho = np.concatenate((MAPbI3_Ortho_CNPbXdata, MAPbI3_Ortho_CHNPbXdata), axis=1)
MAPbI3_Tetra = np.concatenate((MAPbI3_Tetra_CNPbXdata, MAPbI3_Tetra_CHNPbXdata), axis=1)

frc_suball = np.concatenate((MAPbBr3_Ortho, MAPbCl3_Ortho, MAPbI3_Cubic, MAPbI3_Ortho, MAPbI3_Tetra))
frc_suball = np.reshape(frc_suball, (2500, 23*3*10*2))

frc_rest = np.concatenate((MAPbBr3_Cubic, MAPbBr3_Tetra, MAPbCl3_Cubic, MAPbCl3_Tetra))
frc_rest = np.reshape(frc_rest, (2000, 23*3*10*2))

"""
### CHNPbX Features

frc_CHNPbX = np.concatenate((MAPbBr3_Cubic_CHNPbXdata, MAPbBr3_Ortho_CHNPbXdata, MAPbBr3_Tetra_CHNPbXdata, MAPbCl3_Cubic_CHNPbXdata, MAPbCl3_Ortho_CHNPbXdata, MAPbCl3_Tetra_CHNPbXdata, MAPbI3_Cubic_CHNPbXdata, MAPbI3_Ortho_CHNPbXdata, MAPbI3_Tetra_CHNPbXdata))
frc_CHNPbX = np.reshape(frc_CHNPbX, (4500, 11*3*10))

pca = PCA(n_components=2)
values = pca.fit_transform(frc_CHNPbX)
print(pca.explained_variance_ratio_)

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
plt.savefig("MAPbX3_Classification_CHNPbX_PCA.png", dpi=200)
#plt.show()

#pca50 = PCA(n_components=5)
#values = pca50.fit_transform(frc_CHNPbX)
#print(np.sum(pca50.explained_variance_ratio_))

values = umap.UMAP(min_dist=1).fit_transform(frc_CHNPbX)
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
plt.savefig("MAPbX3_Classification_CHNPbX_UMAP.png", dpi=200)
#plt.show()

#values = TSNE(n_components=2, method='exact', metric='euclidean').fit_transform(frc_all)
#pca50 = PCA(n_components=5)
#values = pca50.fit_transform(frc_CHNPbX)
#print(np.sum(pca50.explained_variance_ratio_))

values = TSNE(n_components=2, perplexity=50, verbose=1).fit_transform(frc_CHNPbX)
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
plt.savefig("MAPbX3_Classification_CHNPbX_tSNE.png", dpi=200)
#plt.show()

"""

### Combined Features

frc_all = np.concatenate((MAPbBr3_Cubic, MAPbBr3_Ortho, MAPbBr3_Tetra, MAPbCl3_Cubic, MAPbCl3_Ortho, MAPbCl3_Tetra, MAPbI3_Cubic, MAPbI3_Ortho, MAPbI3_Tetra))
frc_all = np.reshape(frc_all, (4500, 23*3*10*2))
y = []
for i in range(9):
    for j in range(500):
        y.append(i)
y = np.array(y)

sio.savemat("frc.mat", {'fdata':frc_all})

"""
n_splits = 10
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(frc_all):
    X_train, X_test = frc_all[train_index], frc_all[test_index]
    y_train, y_test = y[train_index], y[test_index]

    knn = KNeighborsClassifier().fit(X_train, y_train)
    print(knn.score(X_test, y_test))


pca = PCA(n_components=4)
values = pca.fit_transform(frc_all)
print(pca.explained_variance_ratio_)

fig = plt.figure(figsize=(12,12), dpi=200)
ax = fig.add_subplot(111, projection="3d")
markers = ['^', '^', '^', 's', 's', 's', 'o', 'o', 'o']
idx = 0
for label, col, mm in zip(ligs, colors, markers):
    Axes3D.scatter(ax, -values[idx*(frd-frs+1):(idx+1)*(frd-frs+1), 0], values[idx*(frd-frs+1):(idx+1)*(frd-frs+1),1], values[idx*(frd-frs+1):(idx+1)*(frd-frs+1), 2], color='white', marker=mm, label=label, edgecolor=col, lw=.5, s=20, alpha=0.75)
    idx += 1
ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.zaxis.labelpad = 20
ax.azim = 60
ax.dist = 10
ax.elev = 30
ax.tick_params(axis="both", which="major", labelsize=20)
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_PCA_3D.png", dpi=200)

pca = PCA(n_components=2)
values = pca.fit_transform(frc_all)
print(pca.explained_variance_ratio_)

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
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_PCA.png", dpi=200)
#plt.show()

#pca50 = PCA(n_components=5)
#values = pca50.fit_transform(frc_all)
#print(np.sum(pca50.explained_variance_ratio_))

values = umap.UMAP(min_dist=.5,
    n_components=2,
    random_state=42).fit_transform(frc_all)
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

plt.ylim(np.min(values[:, 1])-10, np.max(values[:,1])+20)
#plt.xlim(-80, 80)
plt.legend(ncol=3, loc='upper left', handlelength=.25, borderpad=.25, fontsize=10)
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_UMAP.png", dpi=200)
#plt.show()

n_splits = 10
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(values):
    X_train, X_test = values[train_index], values[test_index]
    y_train, y_test = y[train_index], y[test_index]

    knn = KNeighborsClassifier().fit(X_train, y_train)
    print(knn.score(X_test, y_test))

h = .02 # step size in the mesh
X = values; Y=y
knn=KNeighborsClassifier()
# we create an instance of Neighbours Classifier and fit the data.
knn.fit(X, Y)
# Plot the decision boundary. For that, we will asign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.set_cmap(plt.cm.Paired)
plt.pcolormesh(xx, yy, Z)
# Plot also the training points
plt.scatter(values[:frd-frs+1,0], values[:frd-frs+1,1], marker='^', color='white', alpha=0.75, edgecolor='tab:blue', linewidth=.5, s=20, label="Br-Cubic")
plt.scatter(values[frd-frs+1:2*(frd-frs+1),0], values[frd-frs+1:2*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:orange', linewidth=0.5, s=20, label="Br-Ortho")
plt.scatter(values[2*(frd-frs+1):3*(frd-frs+1),0], values[2*(frd-frs+1):3*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:green', linewidth=0.5, s=20, label="Br-Tetra")

plt.scatter(values[3*(frd-frs+1):4*(frd-frs+1),0], values[3*(frd-frs+1):4*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:red', linewidth=0.5, s=20, label="Cl-Cubic")
plt.scatter(values[4*(frd-frs+1):5*(frd-frs+1),0], values[4*(frd-frs+1):5*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:purple', linewidth=0.5, s=20, label="Cl-Ortho")
plt.scatter(values[5*(frd-frs+1):6*(frd-frs+1),0], values[5*(frd-frs+1):6*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:brown', linewidth=0.5, s=20, label="Cl-Tetra")

plt.scatter(values[6*(frd-frs+1):7*(frd-frs+1),0], values[6*(frd-frs+1):7*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:pink', linewidth=0.5, s=20, label="I-Cubic")
plt.scatter(values[7*(frd-frs+1):8*(frd-frs+1),0], values[7*(frd-frs+1):8*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:gray', linewidth=0.5, s=20, label="I-Ortho")
plt.scatter(values[8*(frd-frs+1):9*(frd-frs+1),0], values[8*(frd-frs+1):9*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:olive', linewidth=0.5, s=20, label="I-Tetra")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_kNN.png", dpi=200)



#values = TSNE(n_components=2, method='exact', metric='euclidean').fit_transform(frc_all)

#pca50 = PCA(n_components=50)
#values = pca50.fit_transform(frc_all)
#print(np.sum(pca50.explained_variance_ratio_))

values = TSNE(n_components=3, verbose=1).fit_transform(frc_all)

fig = plt.figure(figsize=(12,12), dpi=200)
ax = fig.add_subplot(111, projection="3d")
markers = ['^', '^', '^', 's', 's', 's', 'o', 'o', 'o']
idx = 0
for label, col, mm in zip(ligs, colors, markers):
    Axes3D.scatter(ax, -values[idx*(frd-frs+1):(idx+1)*(frd-frs+1), 0], values[idx*(frd-frs+1):(idx+1)*(frd-frs+1),1], values[idx*(frd-frs+1):(idx+1)*(frd-frs+1), 2], color='white', marker=mm, label=label, edgecolor=col, lw=.5, s=10, alpha=0.75)
    idx += 1
ax.xaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.zaxis.labelpad = 20
ax.azim = 60
ax.dist = 10
ax.elev = 30
ax.tick_params(axis="both", which="major", labelsize=20)
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_tSNE_3D.png", dpi=200)
"""

pca50 = PCA(n_components=50)
values = pca50.fit_transform(frc_all)
print(np.sum(pca50.explained_variance_ratio_))

values = TSNE(n_components=2, verbose=2, learning_rate=375, early_exaggeration=4.0, perplexity=50).fit_transform(values)

plt.figure(figsize=(5,5), dpi=200)
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.scatter(values[:frd-frs,0], values[:frd-frs,1], marker='.', color='tab:blue', alpha=0.75, linewidth=.5, s=20, label="Br-Cubic")
plt.scatter(values[frd-frs:2*(frd-frs),0], values[frd-frs:2*(frd-frs),1], marker='.', color='tab:orange', alpha=0.75,  linewidth=0.5, s=20, label="Br-Ortho")
plt.scatter(values[2*(frd-frs):3*(frd-frs),0], values[2*(frd-frs):3*(frd-frs),1], marker='.', color='tab:green', alpha=0.75,  linewidth=0.5, s=20, label="Br-Tetra")

plt.scatter(values[3*(frd-frs):4*(frd-frs),0], values[3*(frd-frs):4*(frd-frs),1], marker='.', color='tab:red', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Cubic")
plt.scatter(values[4*(frd-frs):5*(frd-frs),0], values[4*(frd-frs):5*(frd-frs),1], marker='.', color='tab:purple', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Ortho")
plt.scatter(values[5*(frd-frs):6*(frd-frs),0], values[5*(frd-frs):6*(frd-frs),1], marker='.', color='tab:brown', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Tetra")

plt.scatter(values[6*(frd-frs):7*(frd-frs),0], values[6*(frd-frs):7*(frd-frs),1],  marker='.',color='tab:pink', alpha=0.75,  linewidth=0.5, s=20, label="I-Cubic")
plt.scatter(values[7*(frd-frs):8*(frd-frs),0], values[7*(frd-frs):8*(frd-frs),1],  marker='.',color='tab:gray', alpha=0.75,  linewidth=0.5, s=20, label="I-Ortho")
plt.scatter(values[8*(frd-frs):9*(frd-frs),0], values[8*(frd-frs):9*(frd-frs),1],  marker='.',color='tab:olive', alpha=0.75,  linewidth=0.5, s=20, label="I-Tetra")

#plt.ylim(np.min(values[:, 1])-10, np.max(values[:,1])+50)
#plt.xlim(-100, 100)
#plt.legend(ncol=3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_tSNE.png", dpi=200)
#plt.show()

"""
n_splits = 10
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(values):
    X_train, X_test = values[train_index], values[test_index]
    y_train, y_test = y[train_index], y[test_index]

    knn = KNeighborsClassifier().fit(X_train, y_train)
    print(knn.score(X_test, y_test))


## Separate Clusters
print("OK Clusters!!")

pca = PCA(n_components=2)
values = pca.fit_transform(frc_suball)
print(pca.explained_variance_ratio_)

plt.figure(dpi=200)
plt.scatter(values[0*(frd-frs+1):1*(frd-frs+1),0], values[0*(frd-frs+1):1*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:orange', linewidth=0.5, s=20, label="Br-Ortho")
plt.scatter(values[1*(frd-frs+1):2*(frd-frs+1),0], values[1*(frd-frs+1):2*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:purple', linewidth=0.5, s=20, label="Cl-Ortho")
plt.scatter(values[2*(frd-frs+1):3*(frd-frs+1),0], values[2*(frd-frs+1):3*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:pink', linewidth=0.5, s=20, label="I-Cubic")
plt.scatter(values[3*(frd-frs+1):4*(frd-frs+1),0], values[3*(frd-frs+1):4*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:gray', linewidth=0.5, s=20, label="I-Ortho")
plt.scatter(values[4*(frd-frs+1):5*(frd-frs+1),0], values[4*(frd-frs+1):5*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:olive', linewidth=0.5, s=20, label="I-Tetra")

#plt.ylim(-400, 700)
#plt.xlim(-1000, 1000)
plt.legend(ncol=3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_PCA_suball.png", dpi=200)
#plt.show()

#pca50 = PCA(n_components=5)
#values = pca50.fit_transform(frc_all)
#print(np.sum(pca50.explained_variance_ratio_))

values = umap.UMAP(min_dist=0.25,
    n_components=2,
    random_state=42).fit_transform(frc_suball)
plt.figure(dpi=200)
plt.scatter(values[0*(frd-frs+1):1*(frd-frs+1),0], values[0*(frd-frs+1):1*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:orange', linewidth=0.5, s=20, label="Br-Ortho")
plt.scatter(values[1*(frd-frs+1):2*(frd-frs+1),0], values[1*(frd-frs+1):2*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:purple', linewidth=0.5, s=20, label="Cl-Ortho")
plt.scatter(values[2*(frd-frs+1):3*(frd-frs+1),0], values[2*(frd-frs+1):3*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:pink', linewidth=0.5, s=20, label="I-Cubic")
plt.scatter(values[3*(frd-frs+1):4*(frd-frs+1),0], values[3*(frd-frs+1):4*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:gray', linewidth=0.5, s=20, label="I-Ortho")
plt.scatter(values[4*(frd-frs+1):5*(frd-frs+1),0], values[4*(frd-frs+1):5*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:olive', linewidth=0.5, s=20, label="I-Tetra")

#plt.ylim(-10, 35)
#plt.xlim(-80, 80)
plt.legend(ncol=3, loc='upper left', handlelength=.25, borderpad=.25, fontsize=10)
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_UMAP_suball.png", dpi=200)
#plt.show()

pca50 = PCA(n_components=50)
values = pca50.fit_transform(frc_suball)
print(np.sum(pca50.explained_variance_ratio_))

values = TSNE(n_components=2, verbose=1).fit_transform(values)

plt.figure(dpi=200)
plt.scatter(values[0*(frd-frs+1):1*(frd-frs+1),0], values[0*(frd-frs+1):1*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:orange', linewidth=0.5, s=20, label="Br-Ortho")
plt.scatter(values[1*(frd-frs+1):2*(frd-frs+1),0], values[1*(frd-frs+1):2*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:purple', linewidth=0.5, s=20, label="Cl-Ortho")
plt.scatter(values[2*(frd-frs+1):3*(frd-frs+1),0], values[2*(frd-frs+1):3*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:pink', linewidth=0.5, s=20, label="I-Cubic")
plt.scatter(values[3*(frd-frs+1):4*(frd-frs+1),0], values[3*(frd-frs+1):4*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:gray', linewidth=0.5, s=20, label="I-Ortho")
plt.scatter(values[4*(frd-frs+1):5*(frd-frs+1),0], values[4*(frd-frs+1):5*(frd-frs+1),1], color='white', alpha=0.75, edgecolor='tab:olive', linewidth=0.5, s=20, label="I-Tetra")

#plt.ylim(-100, 140)
#plt.xlim(-100, 100)
plt.legend(ncol=3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_tSNE_suball.png", dpi=200)
#plt.show()



print("Rest of the Clusters!!")
pca = PCA(n_components=2)
values = pca.fit_transform(frc_rest)
print(pca.explained_variance_ratio_)

plt.figure(dpi=200)
plt.scatter(values[:frd-frs+1,0], values[:frd-frs+1,1], marker='^', color='white', alpha=0.75, edgecolor='tab:blue', linewidth=.5, s=20, label="Br-Cubic")
plt.scatter(values[1*(frd-frs+1):2*(frd-frs+1),0], values[1*(frd-frs+1):2*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:green', linewidth=0.5, s=20, label="Br-Tetra")
plt.scatter(values[2*(frd-frs+1):3*(frd-frs+1),0], values[2*(frd-frs+1):3*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:red', linewidth=0.5, s=20, label="Cl-Cubic")
plt.scatter(values[3*(frd-frs+1):4*(frd-frs+1),0], values[3*(frd-frs+1):4*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:brown', linewidth=0.5, s=20, label="Cl-Tetra")

#plt.ylim(-400, 700)
#plt.xlim(-1000, 1000)
plt.legend(ncol=3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_PCA_rest.png", dpi=200)
#plt.show()

#pca50 = PCA(n_components=5)
#values = pca50.fit_transform(frc_all)
#print(np.sum(pca50.explained_variance_ratio_))

values = umap.UMAP(min_dist=0.25,
    n_components=2,
    random_state=42).fit_transform(frc_rest)
plt.figure(dpi=200)
plt.scatter(values[:frd-frs+1,0], values[:frd-frs+1,1], marker='^', color='white', alpha=0.75, edgecolor='tab:blue', linewidth=.5, s=20, label="Br-Cubic")
plt.scatter(values[1*(frd-frs+1):2*(frd-frs+1),0], values[1*(frd-frs+1):2*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:green', linewidth=0.5, s=20, label="Br-Tetra")
plt.scatter(values[2*(frd-frs+1):3*(frd-frs+1),0], values[2*(frd-frs+1):3*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:red', linewidth=0.5, s=20, label="Cl-Cubic")
plt.scatter(values[3*(frd-frs+1):4*(frd-frs+1),0], values[3*(frd-frs+1):4*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:brown', linewidth=0.5, s=20, label="Cl-Tetra")

#plt.ylim(-10, 35)
#plt.xlim(-80, 80)
plt.legend(ncol=3, loc='upper left', handlelength=.25, borderpad=.25, fontsize=10)
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_UMAP_rest.png", dpi=200)
#plt.show()



#values = TSNE(n_components=2, method='exact', metric='euclidean').fit_transform(frc_all)

pca50 = PCA(n_components=50)
values = pca50.fit_transform(frc_rest)
print(np.sum(pca50.explained_variance_ratio_))

values = TSNE(n_components=2, verbose=1).fit_transform(values)

plt.figure(dpi=200)
plt.scatter(values[:frd-frs+1,0], values[:frd-frs+1,1], marker='^', color='white', alpha=0.75, edgecolor='tab:blue', linewidth=.5, s=20, label="Br-Cubic")
plt.scatter(values[1*(frd-frs+1):2*(frd-frs+1),0], values[1*(frd-frs+1):2*(frd-frs+1),1], marker='^', color='white', alpha=0.75, edgecolor='tab:green', linewidth=0.5, s=20, label="Br-Tetra")
plt.scatter(values[2*(frd-frs+1):3*(frd-frs+1),0], values[2*(frd-frs+1):3*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:red', linewidth=0.5, s=20, label="Cl-Cubic")
plt.scatter(values[3*(frd-frs+1):4*(frd-frs+1),0], values[3*(frd-frs+1):4*(frd-frs+1),1], marker='s', color='white', alpha=0.75, edgecolor='tab:brown', linewidth=0.5, s=20, label="Cl-Tetra")

#plt.ylim(-100, 140)
#plt.xlim(-100, 100)
plt.legend(ncol=3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_tSNE_rest.png", dpi=200)
#plt.show()


pca50 = PCA(n_components=50)
values = pca50.fit_transform(frc_all)
print(np.sum(pca50.explained_variance_ratio_))

affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
    values,
    perplexities=[5, 500],
    n_jobs=8,
    random_state = 3
)

init = openTSNE.initialization.pca(values, random_state=42)

embedding_multiscale = openTSNE.TSNE(n_jobs=8, verbose=True, early_exaggeration=4.0, learning_rate=1000).fit(
    affinities=affinities_multiscale_mixture, 
    initialization=init,
)

print(np.shape(embedding_multiscale))

values = embedding_multiscale

plt.figure(figsize=(5,5), dpi=200)
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.scatter(values[:frd-frs,0], values[:frd-frs,1], marker='.', color='tab:blue', alpha=0.75, linewidth=.5, s=20, label="Br-Cubic")
plt.scatter(values[frd-frs:2*(frd-frs),0], values[frd-frs:2*(frd-frs),1], marker='.', color='tab:orange', alpha=0.75,  linewidth=0.5, s=20, label="Br-Ortho")
plt.scatter(values[2*(frd-frs):3*(frd-frs),0], values[2*(frd-frs):3*(frd-frs),1], marker='.', color='tab:green', alpha=0.75,  linewidth=0.5, s=20, label="Br-Tetra")

plt.scatter(values[3*(frd-frs):4*(frd-frs),0], values[3*(frd-frs):4*(frd-frs),1], marker='.', color='tab:red', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Cubic")
plt.scatter(values[4*(frd-frs):5*(frd-frs),0], values[4*(frd-frs):5*(frd-frs),1], marker='.', color='tab:purple', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Ortho")
plt.scatter(values[5*(frd-frs):6*(frd-frs),0], values[5*(frd-frs):6*(frd-frs),1], marker='.', color='tab:brown', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Tetra")

plt.scatter(values[6*(frd-frs):7*(frd-frs),0], values[6*(frd-frs):7*(frd-frs),1],  marker='.',color='tab:pink', alpha=0.75,  linewidth=0.5, s=20, label="I-Cubic")
plt.scatter(values[7*(frd-frs):8*(frd-frs),0], values[7*(frd-frs):8*(frd-frs),1],  marker='.',color='tab:gray', alpha=0.75,  linewidth=0.5, s=20, label="I-Ortho")
plt.scatter(values[8*(frd-frs):9*(frd-frs),0], values[8*(frd-frs):9*(frd-frs),1],  marker='.',color='tab:olive', alpha=0.75,  linewidth=0.5, s=20, label="I-Tetra")

#plt.ylim(np.min(values[:, 1])-10, np.max(values[:,1])+50)
#plt.xlim(-100, 100)
#plt.legend(ncol=3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.savefig("MAPbX3_Classification_CNPbX_CHNPbX_opentSNE.png", dpi=200)
#plt.show()
"""