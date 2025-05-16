
from sklearn.decomposition import PCA
import numpy as np
import os
import joblib
import argparse

parser=argparse.ArgumentParser()

parser.add_argument("--dim",type=int,default=4096)

args=parser.parse_args()
dim=args.dim


subject_class_labels={
    1:np.array([1,0,0,0]),
    2:np.array([0,1,0,0]),
    5:np.array([0,0,1,0]),
    7:np.array([0,0,0,1])
}
sublist=[1, 2, 5, 7]
for sub in sublist:
    train_fmri=[]
    train_img=[]
    train_labels=[]

    test_fmri=[]
    test_img=[]
    test_labels=[]
    fmri_suffix="flattened"
    path=os.environ["BRAIN_DATA_DIR"]+f"/subj{sub:02d}_fmri{fmri_suffix}_stim_paired.npz"
    os.makedirs(os.path.join(os.environ["BRAIN_DATA_DIR"],"pca"),exist_ok=True)
    npz_loaded=np.load(path,allow_pickle=True)
    subject_fmri_train=npz_loaded["fmri_train"]
    subject_stim_train=npz_loaded["stim_train"]
    subject_fmri_test=npz_loaded["fmri_test"]
    subject_stim_test=npz_loaded["stim_test"]

    n_test=len(subject_fmri_test)
    n_train=len(subject_fmri_train)

    subject_train_labels=[subject_class_labels[sub] for _ in range(n_train)]
    subject_test_labels=[subject_class_labels[sub] for _ in range(n_test)]

    
    print(sub,'subject_fmri_train.max(),subject_fmri_train.min()',subject_fmri_train.max(),subject_fmri_train.min())
    print(sub,'subject_stim_train.max(),subject_fmri_train.min()',subject_stim_train.max(),subject_stim_train.min())
    print(sub,'subject_fmri_test.max(),subject_fmri_test.min()',subject_fmri_test.max(),subject_fmri_test.min())
    print(sub,'subject_stim_test,max(),subject_stim_test.min()',subject_stim_test.max(),subject_stim_test.min())
    
    fmri_min=min(subject_fmri_train.min(),subject_fmri_test.min())
    fmri_max=max(subject_fmri_train.max(),subject_fmri_test.max())

    subject_fmri_train=2 * (subject_fmri_train-fmri_min) / (fmri_max-fmri_min) -1
    subject_fmri_test=2 * (subject_fmri_test-fmri_min) / (fmri_max-fmri_min) -1

    train_fmri.extend(subject_fmri_train)
    train_img.extend(subject_stim_train)
    train_labels.extend(subject_train_labels)

    test_fmri.extend(subject_fmri_test)
    test_img.extend(subject_stim_test)
    test_labels.extend(subject_test_labels)


    pca = PCA(n_components=dim)
    pca.fit(train_fmri)

    joblib.dump(pca, os.path.join(os.environ["BRAIN_DATA_DIR"],"pca",f"subj{sub:02d}_{dim}_fmri.pkl"))
    print("saved ",f"subj{sub:02d}_fmri.pkl")
