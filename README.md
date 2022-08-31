# PF-OHIP
This manual contains the models implemented in the paper "Topological Feature Engineering for Machine Learning Based Halide Perovskite Materials Design".

# Code Requirements
---
        Platform: Python>=3.6, MATLAB 2016B
        Python Packages needed: math, numpy>=1.19.5, scipy>=1.4.1, scikit-learn>=0.20.3, GUDHI 3.0.0, GeneralisedFormanRicci==0.3
        
# Pipeline of PF-GBT Models
---
![flowchart](https://user-images.githubusercontent.com/32187437/187708602-27ee23d5-70a4-41e2-a518-0663df03ab0b.png)

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
Both MATLAB and some python code for classification process. Some code snippets included were also used to generate Figure 3(a).

```python
tSNE_Trunc_MAPbX.m --> Actual t-SNE classification code used for PH features. 
MAPbX3_classify_CHNPbX.py --> Additional t-SNE classification using just CHNPbX PH features. 
MAPbX3_classify_CNPbX.py --> Additional t-SNE classification using just CNPbX PH features. 
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
# Bandgap Predictions
Here, we provide the codes for the bandgap prediction of the OIHP structures. 
```python
./TPD_ML_predictions/ --> Folder for the Traditional Perovskite Descriptors ML Model. 
./PH_ML_predictions/ --> Folder for the PH-based GBT Model. 
./FRC_ML_predictions/ --> Folder for the PFRC-based GBT Model.
```

## Cite
If you use this code in your research, please considering cite our paper:

*  D. Vijay Anand, Qiang Xu, JunJie Wee, Kelin Xia, Tze Chien Sum. Topological Feature Engineering for Machine Learning Based Halide Perovskite Materials Design. npj Computational Materials. (2022). (to appear)
