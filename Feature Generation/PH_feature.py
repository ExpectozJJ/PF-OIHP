import numpy as np 
import gudhi as gd
import os 
import glob
import csv

def convertpdb(filename):
    f=open(filename, "r")
    if f.mode == 'r':
        contents = f.readlines()
    
    #recordname = []

    #atomNum = []
    atomName = []
    #altLoc = []
    #resName = []

    #chainID = []
    #resNum = []
    X = []
    Y = []
    Z = []

    #occupancy = []
    #betaFactor = []
    element = []
    #charge = []
    
    
    for i in range(len(contents)):
        thisLine = contents[i]

        if thisLine[0:4]=='ATOM' or thisLine[0:6]=='HETATM':
            #recordname = np.append(recordname,thisLine[:6].strip())
            #atomNum = np.append(atomNum, float(thisLine[6:11]))
            atomName = np.append(atomName, thisLine[12:16])
            #altLoc = np.append(altLoc,thisLine[16])
            #resName = np.append(resName, thisLine[17:20].strip())
            #chainID = np.append(chainID, thisLine[21])
            #resNum = np.append(resNum, float(thisLine[23:26]))
            X = np.append(X, float(thisLine[30:38]))
            Y = np.append(Y, float(thisLine[38:46]))
            Z = np.append(Z, float(thisLine[46:54]))
            #occupancy = np.append(occupancy, float(thisLine[55:60]))
            #betaFactor = np.append(betaFactor, float(thisLine[61:66]))
            element = np.append(element,thisLine[12:15])

    #print(atomName)
    a = {'PRO': [{'atom': atomName, 'typ': element, 'pos': np.transpose([X,Y,Z])}]}
    np.savez(filename[:-4]+".npz", **a)

os.chdir("/Data/")

filenames = []
for file in glob.glob("*.pdb"): # Getting all the pdb files of supercell
    filenames.append(file)

for i in range(len(filenames)):
    convertpdb(filenames[i])
    print("Converted {}, {}".format(i, filenames[i]))

np.save("../compoundname.npy", filenames)

dataset = []
for i in range(len(filenames)):
    Asite_dict = {'C':[], 'N':[], 'H':[], 'O':[]}
    Bsite_dict = {'Ge':[], 'Pb':[], 'Sn':[]}
    Xsite_dict = {'Br':[], 'Cl':[], 'F':[], 'I':[]}
    tmp = np.load(filenames[i][:-4]+".npz", allow_pickle=True)
    tmp = tmp["PRO"]
    for t in tmp:
        atmtyp = t["atom"]
        coords = t["pos"]
    for j in range(len(atmtyp)):
        if atmtyp[j].strip(" ") in Asite_dict.keys():
            Asite_dict[atmtyp[j].strip(" ")].append(coords[j])
        if atmtyp[j].strip(" ") in Bsite_dict.keys():
            Bsite_dict[atmtyp[j].strip(" ")].append(coords[j])
        if atmtyp[j].strip(" ") in Xsite_dict.keys():
            Xsite_dict[atmtyp[j].strip(" ")].append(coords[j])
    dataset.append([Asite_dict, Bsite_dict, Xsite_dict])

ph_input = []
comb1 = [['C'], ['N'], ['H'], ['O'], ['B'], ['X']]
comb2 = [['B', 'X'], ['C', 'B'], ['H', 'B'], ['N', 'B'], ['O', 'B'], ['C', 'X'], ['H', 'X'], ['N', 'X'], ['O', 'X']]
comb3 = [['C', 'B', 'X'], ['H', 'B', 'X'], ['N', 'B', 'X'], ['O', 'B', 'X']]
comb4 = [['C', 'H', 'B', 'X'], ['C', 'N', 'B', 'X'], ['C', 'O', 'B', 'X'], ['H', 'N', 'B', 'X'], ['H', 'O', 'B', 'X'], ['N', 'O', 'B', 'X']]
comb5 = [['C', 'N', 'H', 'B', 'X'], ['C', 'O', 'H', 'B', 'X'], ['O', 'N', 'H', 'B', 'X'], ['C', 'N', 'O', 'B', 'X']]
comb6 = [['C', 'N', 'O', 'H', 'B', 'X']]

for i in range(len(dataset)):
    ptclouds = []
    [Asite_dict, Bsite_dict, Xsite_dict] = dataset[i]
    for j in range(len(comb1)):
        pts = []
        for k in comb1[j]:
            if k in Asite_dict.keys():
                for l in range(len(Asite_dict[k])):
                    pts.append(Asite_dict[k][l])
            if k == 'B':
                for key, val in Bsite_dict.items():
                    for l in range(len(Bsite_dict[key])):
                        pts.append(Bsite_dict[key][l])
            if k == 'X':
                for key, val in Xsite_dict.items():
                    for l in range(len(Xsite_dict[key])):
                        pts.append(Xsite_dict[key][l])
        ptclouds.append(pts)
        
    for j in range(len(comb2)):
        pts = []
        for k in comb2[j]:
            if k in Asite_dict.keys():
                for l in range(len(Asite_dict[k])):
                    pts.append(Asite_dict[k][l])
            if k == 'B':
                for key, val in Bsite_dict.items():
                    for l in range(len(Bsite_dict[key])):
                        pts.append(Bsite_dict[key][l])
            if k == 'X':
                for key, val in Xsite_dict.items():
                    for l in range(len(Xsite_dict[key])):
                        pts.append(Xsite_dict[key][l])
                        
        ptclouds.append(pts)
        
    for j in range(len(comb3)):
        pts = []
        for k in comb3[j]:
            if k in Asite_dict.keys():
                for l in range(len(Asite_dict[k])):
                    pts.append(Asite_dict[k][l])
            if k == 'B':
                for key, val in Bsite_dict.items():
                    for l in range(len(Bsite_dict[key])):
                        pts.append(Bsite_dict[key][l])
            if k == 'X':
                for key, val in Xsite_dict.items():
                    for l in range(len(Xsite_dict[key])):
                        pts.append(Xsite_dict[key][l])
                        
        ptclouds.append(pts)
        
    for j in range(len(comb4)):
        pts = []
        for k in comb4[j]:
            if k in Asite_dict.keys():
                for l in range(len(Asite_dict[k])):
                    pts.append(Asite_dict[k][l])
            if k == 'B':
                for key, val in Bsite_dict.items():
                    for l in range(len(Bsite_dict[key])):
                        pts.append(Bsite_dict[key][l])
            if k == 'X':
                for key, val in Xsite_dict.items():
                    for l in range(len(Xsite_dict[key])):
                        pts.append(Xsite_dict[key][l])
                        
        ptclouds.append(pts)
        
    for j in range(len(comb5)):
        pts = []
        for k in comb5[j]:
            if k in Asite_dict.keys():
                for l in range(len(Asite_dict[k])):
                    pts.append(Asite_dict[k][l])
            if k == 'B':
                for key, val in Bsite_dict.items():
                    for l in range(len(Bsite_dict[key])):
                        pts.append(Bsite_dict[key][l])
            if k == 'X':
                for key, val in Xsite_dict.items():
                    for l in range(len(Xsite_dict[key])):
                        pts.append(Xsite_dict[key][l])
                        
        ptclouds.append(pts)
        
    for j in range(len(comb6)):
        pts = []
        for k in comb6[j]:
            if k in Asite_dict.keys():
                for l in range(len(Asite_dict[k])):
                    pts.append(Asite_dict[k][l])
            if k == 'B':
                for key, val in Bsite_dict.items():
                    for l in range(len(Bsite_dict[key])):
                        pts.append(Bsite_dict[key][l])
            if k == 'X':
                for key, val in Xsite_dict.items():
                    for l in range(len(Xsite_dict[key])):
                        pts.append(Xsite_dict[key][l])
                        
        ptclouds.append(pts)
    
    ph_input.append(ptclouds)

np.save("../ph_input.npy", ph_input)
inp = np.load("../ph_input.npy", allow_pickle=True)

bars = []
for i in range(len(filenames)):
    print(filenames[i])
    code = []
    for j in range(len(inp[i])):
        if j<= 5:
            sc = gd.RipsComplex(inp[i][j], max_edge_length = 15)
        else:
            sc = gd.RipsComplex(inp[i][j], max_edge_length = 10)
        st = sc.create_simplex_tree(max_dimension=2)
        dgmsalpha = st.persistence()
        code.append(dgmsalpha)
    bars.append(code)

np.save("../PH_betti_v3.npy", bars)

all_feat = []
for i in range(len(bars)):
    feat = []
    for j in range(len(bars[i])):
        dgmsalpha = bars[i][j]
        betti0, betti1 = [], []
        for r in dgmsalpha:
            if r[0] == 0:
                betti0.append([r[1][0], r[1][1]])
            elif r[0] == 1:
                betti1.append([r[1][0], r[1][1]])

        betti0 = np.array(betti0)
        betti1 = np.array(betti1)
        betti = [betti0, betti1]

        betti0 = sorted(betti[0], key=lambda x: x[0])
        betti0 = np.flip(betti0, axis=0)
        betti1 = sorted(betti[1], key=lambda x: x[0])
        betti1 = np.flip(betti1, axis=0)
        
        #print(np.shape(betti0), np.shape(betti1))
        
        pb0, pb1 = np.zeros(100), np.zeros(100)
        
        if j <= 5:
            idx = np.linspace(0, 15, 101)
            for k in range(len(idx)-1):
                for l in range(len(betti0)):
                    if betti0[l][0] <= idx[k] and betti0[l][1]>= idx[k+1]:
                        pb0[k]+=1
                for l in range(len(betti1)):
                    if betti1[l][0] <= idx[k] and betti1[l][1]>= idx[k+1]:
                        pb1[k]+=1
        else:
            idx = np.linspace(0, 10, 101)
            for k in range(len(idx)-1):
                for l in range(len(betti0)):
                    if betti0[l][0] <= idx[k] and betti0[l][1]>= idx[k+1]:
                        pb0[k]+=1
                for l in range(len(betti1)):
                    if betti1[l][0] <= idx[k] and betti1[l][1]>= idx[k+1]:
                        pb1[k]+=1
                        
        feat.append([pb0, pb1])
    #print(np.shape(feat))
    all_feat.append(feat)

all_feat = np.array(all_feat)
np.save("../PBN_320.npy", all_feat)
