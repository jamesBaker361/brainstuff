import numpy as np

file=np.load("/umbc/ada/donengel/common/brain/data/subj01_fmriflattened_stim_paired.npz")

for i,img in enumerate(file["stim_test"]):
    if img.shape!=(425,425,3):
        print(f" array img {i} == ",img.shape)



file=np.load("/umbc/ada/donengel/common/brain/data/subj01_fmri_stim_paired.npz")

for i,img in enumerate(file["stim_test"]):
    if img.shape!=(425,425,3):
        print(f" voxel img {i} == ",img.shape)

