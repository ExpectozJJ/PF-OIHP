import numpy as np
import math
import scipy.io as sio
from GeneralisedFormanRicci.frc import GeneralisedFormanRicci, n_faces
import os
import sys
import umap
import umap.plot
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

def normalize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    length1 = X.shape[0]
    X_train_normed = X

    for i in range(0,length1):
        for j in range(0,X.shape[1]):
            if std[j]!=0 :
                X_train_normed[i,j] = (X_train_normed[i,j]-mean[j])/std[j]
    return X_train_normed

MAPbBr3_Cubic_CNPbXdata = []
MAPbCl3_Cubic_CNPbXdata = []
MAPbI3_Cubic_CNPbXdata = []
MAPbBr3_Ortho_CNPbXdata = []
MAPbCl3_Ortho_CNPbXdata = []
MAPbI3_Ortho_CNPbXdata = []
MAPbBr3_Tetra_CNPbXdata = []
MAPbCl3_Tetra_CNPbXdata = []
MAPbI3_Tetra_CNPbXdata = []

print("Archiving Features...")
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

frs = 500
frd = 1000

frc_all = np.concatenate((MAPbBr3_Cubic_CNPbXdata, MAPbBr3_Ortho_CNPbXdata, MAPbBr3_Tetra_CNPbXdata, MAPbCl3_Cubic_CNPbXdata, MAPbCl3_Ortho_CNPbXdata, MAPbCl3_Tetra_CNPbXdata, MAPbI3_Cubic_CNPbXdata, MAPbI3_Ortho_CNPbXdata, MAPbI3_Tetra_CNPbXdata))
frc_all = np.reshape(frc_all, (4500, 11, 3, 10))
frc_all = np.concatenate((frc_all[:, 2:7, 0, :], frc_all[:, 2:7, 1, :], frc_all[:, 2:7, 2, :]), axis=1)
frc_all = np.reshape(frc_all, (4500, 5*3*10))

#pca = KernelPCA(n_components=3)
#values = pca.fit_transform(frc_all)
#print(pca.explained_variance_ratio_)

pca = KernelPCA(n_components=2)
values = pca.fit_transform(frc_all)
#print(pca.explained_variance_ratio_)

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



values = TSNE(n_components=2, n_iter=500, perplexity=50).fit_transform(frc_all)
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

