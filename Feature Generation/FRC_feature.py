from numpy import loadtxt
import numpy as np 
from GeneralisedFormanRicci.frc import GeneralisedFormanRicci, n_faces
import math
import time
import sys
import multiprocessing as mp 
from sklearn import ensemble
import os 
import glob

def compute_stat(dist): # Compute Statistical Features with input FRC distribution dist
    #print(dist)
    dist = np.array(dist)
    if len(dist) == 0:
        feat = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] # If the input is empty, output all-zero stat
    else:
        feat = []
        feat.append(np.min(dist))                       # FPRC Minimum
        feat.append(np.max(dist))                       # FPRC Maximum
        feat.append(np.mean(dist))                      # FPRC Mean
        feat.append(np.std(dist))                       # FPRC Standard Deviation
        pos_sum, pos2, u, v = 0, 0, [], 0
        for l in dist:
            if l > 0:
                pos_sum += l
                pos2 += l*l 
                u.append(abs(l-feat[2]))
                v += 1/l
        
        u = np.array(u)
        feat.append(pos_sum)                             # FPRC positive sum 
        feat.append(np.sum(u))                           # FPRC Absolute Deviation (PerORC Generalised Graph Energy) 
        feat.append(np.sum(dist*dist))                   # FPRC Simplicial Complex energy 2nd moment
        feat.append(pos2)                                # FPRC positive sum 2nd moment
        feat.append(math.log(len(dist)*v+1))             # FPRC Quasi-Wiener Index
        feat.append(np.sum(u*u*u))                       # FPRC Absolute Deviation 3rd Moment

    return feat

os.chdir("./structures/")

filenames = []
idx = []
i = 0 
for file in glob.glob("*.pdb"):
    filenames.append(file)
    idx.append(i); i+=1

def PFRC(filename, i):
    print(filename, i)
    inp = np.load("../ph_input.npy", allow_pickle=True)
    code = []
    for j in range(len(inp[i])):
        print(filename, j)
        out = []
        if j<= 5:
            for f in np.arange(0, 15.1, 1):
                #print(f)
                sc = GeneralisedFormanRicci(points=inp[i][j], epsilon=f, method="alpha", p=2)
                frc = sc.compute_forman()
                features = []
                for key in range(3):
                    try:
                        val = frc[key]
                    except:
                        val = {} #Take input as empty dictionary since there are no p-dimensional simplex
                    features.append(compute_stat(list(val.values()))) # For every p-dimensional simplices, compute the statistical features.
                out.append(features)
        else:
            for f in np.arange(0, 10.1, 1):
                #print(f)
                sc = GeneralisedFormanRicci(points=inp[i][j], epsilon=f, method="alpha", p=2)
                frc = sc.compute_forman()
                features = []
                for key in range(3):
                    try:
                        val = frc[key]
                    except:
                        val = {} #Take input as empty dictionary since there are no p-dimensional simplex
                    features.append(compute_stat(list(val.values()))) # For every p-dimensional simplices, compute the statistical features.
                out.append(features)
        code.append(out)
    
    feat = code[0]
    for i in range(1, len(code)):
        feat = np.concatenate((feat, code[i]))
    
    return feat

no_threads = mp.cpu_count()
p = mp.Pool(processes = no_threads)
results = p.starmap(PFRC, zip(filenames, idx))
#results = p.map(func, arr)
p.close()
p.join()

np.save("FRC_320.npy", results)
print(np.shape(results))
