import numpy as np
import math
import scipy.io as sio
from GeneralisedFormanRicci.frc import GeneralisedFormanRicci, n_faces
import time
import sys
import multiprocessing as mp 

# gen_graph generates a graph network (0-simplex and 1-simplex) of the simplicial complex.
# n_faces outputs all the simplices for the simplicial complex for the given filtration parameter.


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

start = time.time()
ndata = 1
maxfiltration = 10
step=0.25
fact=int(1.0/step)
fmax= 7.25
fid = 25;

#begin, last, folder = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
folder = sys.argv[1]

def func(idx):
    print(idx)
    statdata = [[ 0 for i in range(fid) ] for j in range(ndata) ]
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
        for f in np.arange(1, 6.75, 0.25):
            print(idx, f)
            sc = GeneralisedFormanRicci(points=contents, epsilon=f, method="alpha", p=2)
            frc = sc.compute_forman()
            features = []
            for key in range(3):
                try:
                    val = frc[key]
                except:
                    val = {} #Take input as empty dictionary since there are no p-dimensional simplex
                features.append(compute_stat(list(val.values()))) # For every p-dimensional simplices, compute the statistical features.
            out.append(features)
            #print(out)
        statdata[ii] = out
        # print(fno)
    fdata = np.reshape(statdata, (ndata, 23*3*10))
    np.savez('./'+folder+"/CNXPb_atmlist_L5_f{}_frcstat.npz".format(idx),  fdata = fdata)

"""
no_threads = mp.cpu_count()
p = mp.Pool(processes = no_threads)
results = p.map(func, list(range(begin, last)))
#results = p.map(func, arr)
p.close()
p.join()
"""

#arr = np.concatenate((range(603, 608), range(625, 633), range(666, 677), range(728, 731), range(780, 794))) # Cl3 Cubic CNXPb
arr = np.concatenate((range(828, 835), range(860, 865), range(921, 922), range(956, 960), range(993, 996))) #I3 Cubic CNXPb
for val in arr:
    func(val)

        
        
        
        
        
        