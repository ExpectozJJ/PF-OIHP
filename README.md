# PF-OHIP
This manual contains the models implemented in the paper "Topological Feature Engineering for Machine Learning Based Halide Perovskite Materials Design".

# Code Requirements
---
        Platform: Python>=3.6, MATLAB 2016B
        Python Packages needed: math, numpy>=1.19.5, scipy>=1.4.1, scikit-learn>=0.20.3, GUDHI 3.0.0, GeneralisedFormanRicci==0.3
        
# Pipeline of PF-GBT Models
---
![image](https://user-images.githubusercontent.com/32187437/147686016-da8e2f85-7de9-47c1-b7a5-8cabb1a4635b.png)

# Feature Generation

## Persistent Homology

Here, we provide two possible Feature extraction codes. One is written in python and one is written in MATLAB. 

```python
PrepareHOIPFeatures.m --> Extracts the betti numbers from the 30 atom sets barcode informations (MATLAB Feature Extraction)
PH_feature.py --> Generates the 30 atom set barcodes from file directory of all *.pdb files storing atom coordinates for OIHP structures. 

```
## Persistent Forman Ricci Curvature
Note that the code for Persistent Forman Ricci curvature requires GUDHI 3.0.0 and GeneralisedFormanRicci v0.3. 
```python
FRC_feature.py --> Generates all the FRC values from file ph_input.npy storing atom coordinates for all the necessary atom sets in OIHP structures. 
```

# Classification 
Here, we provide the t-SNE codes for the classifications of 9 types of OIHP structures. 

## Persistent Homology
The last 100 frames for MAPbBr3 Cubic structures can be found in ./PH_OUT_MAPbBr3_Cubic_CNPbX/ folder. 
```python
FRC_alpha_gen_CNPbX.py --> The parallel processing code for generating all the FRC values from 4500 OIHP molecular dynamic simulation frames. 
FRC_alpha_CHNPbX.py --> Code to compute the statistical attributes or the FRC feature vectors from the OIHP structures (including hydrogen atoms).
FRC_alpha_CNPbX.py --> Code to compute the statistical attributes or the FRC feature vectors from the OIHP structures (excluding hydrogen atoms).
FRC_alpha_classify_CHNPbX.py --> t-SNE classification using just CHNPbX features. 
FRC_alpha_classify_CNPbX.py --> t-SNE classification using just CNPbX features.
FRC_alpha_classify_CNPbX_CHNPbX.py --> t-SNE classification using both CHNPbX and CNPbX features.
```

## Persistent Forman Ricci Curvature 
The features for MAPb<X-Site>3_<type>_CNXPb_data_feat.npy have been provided where <X-site> = Br, Cl or I and <type> = Cubic, Orthorhombic or Tetragonal. 
```python
FRC_alpha_gen_CNPbX.py --> The parallel processing code for generating all the FRC values from 4500 OIHP molecular dynamic simulation frames. 
FRC_alpha_CHNPbX.py --> Code to compute the statistical attributes or the FRC feature vectors from the OIHP structures (including hydrogen atoms).
FRC_alpha_CNPbX.py --> Code to compute the statistical attributes or the FRC feature vectors from the OIHP structures (excluding hydrogen atoms).
FRC_alpha_classify_CHNPbX.py --> t-SNE classification using just CHNPbX features. 
FRC_alpha_classify_CNPbX.py --> t-SNE classification using just CNPbX features.
FRC_alpha_classify_CNPbX_CHNPbX.py --> t-SNE classification using both CHNPbX and CNPbX features.
```


## Cite
If you use this code in your research, please considering cite our paper:

*  D. Vijay Anand, Qiang Xu, JunJie Wee, Kelin Xia, Tze Chien Sum. Topological Feature Engineering for Machine Learning Based Halide Perovskite Materials Design. <span style="font-style: italic;">npj Computational Materials</span>. (2022). (to appear)


        




## Persistent Spectral (PerSpect) representation and Feature generation

For each protein-protein complex, the relevant element specific atom groups are used to construct the simplicial complexes to generate the Hodge Laplacians. 
```python
def hodge(args):
    # this function generates the eigenvalues from the L0 hodge laplacians in wild and mutation types from Vietoris Rips Complexes. 
    # parameter "args" is foldername format (PDBID, chainid, wildtype, resid, mutanttype) e.g. (1AK4 D A 488 G)
    # Runs the "binding_hodge.m" and "mutation_hodge.m" to generate the hodge laplacians for each complex. 
    
def alpha(args):
    # this function generates the eigenvalues from the L1 and L2 hodge laplacians in wild and mutation types from alpha complexes.
    # parameter "args" is foldername format (PDBID, chainid, wildtype, resid, mutanttype) e.g. (1AK4 D A 488 G)
    # Outputs numpy array of eigenvalues which can be used to generate persistent attributes.
 
def read_eig(args):
    # this function takes the eigenvalues from "alpha" and "hodge" functions and compute the persistent spectral attributes. 
    # parameter "args" is foldername format (PDBID, chainid, wildtype, resid, mutanttype) e.g. (1AK4 D A 488 G)
    # Outputs the persistent attributes in a numpy array after passing through the subfunction "compute_stat"

def compute_stat(dist):
    # this function takes a distribution of eigenvalues of a particular filtration parameter and computes the persistent spectral attributes. 
    # Outputs the persistent attributes of that filtration parameter. 
```
The above functions can be found in feature/skempi_feature.py and feature/run_all.py.

```python
binding_hodge.m --> Computes L0 Hodge Laplacian by constructing Vietoris Rips Complex from the atom coordinates between the binding sites. 
mutation_hodge.m --> Computes L0 Hodge Laplacian by constructing Vietoris Rips Complex from the atom coordinates between the mutation site and its neighborhood.
computeVRcomplex.m --> Constructs the VR complex from atom coordinates.
```

## Base Learner Training
---
Each persistent attribute is put into a base learner (1D-CNN) for training. 

  spectcnn_ab_prelu_v6 --> Base Learner for AB-Bind S645 (with non-binders)
  
  spectcnn_ab_prelu_Nout_v6 --> Base Learner for AB-Bind S645 (without non-binders)
  
  spectcnn_skempi_prelu_v6 --> Base Learner for SKEMPI 1131
  
  spectcnn_homology_test --> Base Learner for AB-Bind S645 Blind Homology Test Prediction
  
## Meta Learner Training 
--- 
The outputs from Base Learners are then combined and input into the Meta Learner for final prediction. 

  spectnettree_ab_prelu_v6 --> Meta Learner for AB-Bind S645 (with non-binders)
  
  spectnettree_ab_prelu_Nout_v6 --> Meta Learner for AB-Bind S645 (without non-binders)
  
  spectnettree_skempi_prelu_v6 --> Meta Learner for SKEMPI 1131
  
  spectnettree_homology_test --> Meta Learner for AB-Bind S645 Blind Homology Test Prediction

# Auxiliary Features
---     
        Software: Jackal, PDB2PQR, SPIDER2, MIBPB
        Relevant code and software can be found in https://doi.org/10.24433/CO.0537487.v1. 
        The auxiliary features for AB-Bind S645 is in models/X_ab_aux.npy.
        The auxiliary features for SKEMPI-1131 is in models/X_skempi_aux.npy.

## Cite
If you use this code in your research, please cite our paper:

* JunJie Wee and Kelin Xia, "Persistent spectral based ensemble learning (PerSpect-EL) for protein-protein binding affinity prediction", Briefings In Bioinformatics, bbac024 (2022)
