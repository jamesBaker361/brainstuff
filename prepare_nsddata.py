import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)



stim_order_f = os.environ["BRAIN_DATA_DIR"]+'/nsddata/experiments/nsd/nsd_expdesign.mat'
stim_order = loadmat(stim_order_f)


## Selecting ids for training and test data

sig_train = {}
sig_test = {}
num_trials = 37*750
for idx in range(num_trials):
    ''' nsdId as in design csv files'''
    nsdId = stim_order['subjectim'][sub-1, stim_order['masterordering'][idx] - 1] - 1
    if stim_order['masterordering'][idx] > 1000:
        sig_train.setdefault(nsdId, []).append(idx)
    else:
        sig_test.setdefault(nsdId, []).append(idx)


train_im_idx = list(sig_train.keys())
test_im_idx = list(sig_test.keys())


roi_dir = os.environ["BRAIN_DATA_DIR"]+'/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub)
betas_dir = os.environ["BRAIN_DATA_DIR"]+'/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub)

mask_filename = 'nsdgeneral.nii.gz'
mask = nib.load(roi_dir+mask_filename).get_fdata()
num_voxel = mask[mask>0].shape[0]

fmri = np.zeros((num_trials, num_voxel)).astype(np.float32)
for i in range(37):
    beta_filename = "betas_session{0:02d}.nii.gz".format(i+1)
    beta_f = nib.load(betas_dir+beta_filename).get_fdata().astype(np.float32)
    fmri[i*750:(i+1)*750] = beta_f[mask>0].transpose()
    if i <3:
        print(i)
        print('beta_f.shape',beta_f.shape)
        print('beta_f[mask>0].transpose()',beta_f[mask>0].transpose())
    del beta_f
    
    
print("fMRI Data are loaded.")

f_stim = h5py.File(os.environ["BRAIN_DATA_DIR"]+'/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
stim = f_stim['imgBrick'][:]

print("Stimuli are loaded.",stim.shape)

num_train, num_test = len(train_im_idx), len(test_im_idx)
vox_dim, im_dim, im_c = num_voxel, 425, 3
fmri_train = np.zeros((num_train,vox_dim))
stim_train = np.zeros((num_train,im_dim,im_dim,im_c))
for i,idx in enumerate(train_im_idx):
    stim_train[i] = stim[idx]
    fmri_train[i] = fmri[sorted(sig_train[idx])].mean(0)
    if i <3:
        print(f"{i}/{len(train_im_idx)}")
        print('fmri_train[i].shape',fmri_train[i].shape)
        print('sorted(sig_train[idx])',sorted(sig_train[idx]))

#np.save(os.environ["BRAIN_DATA_DIR"]+'/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub),fmri_train )
#np.save(os.environ["BRAIN_DATA_DIR"]+'/processed_data/subj{:02d}/nsd_train_stim_sub{}.npy'.format(sub,sub),stim_train )

print("Training data is saved.")

fmri_test = np.zeros((num_test,vox_dim))
stim_test = np.zeros((num_test,im_dim,im_dim,im_c))
for i,idx in enumerate(test_im_idx):
    stim_test[i] = stim[idx]
    fmri_test[i] = fmri[sorted(sig_test[idx])].mean(0)
    if i <3:
        print("stim_test[i].shape",stim_test[i].shape)
        print('fmri_test[i].shape',fmri_test[i].shape)
        print('sorted(sig_test[idx])',sorted(sig_test[idx]))
        print(f"Test {i+1}/{num_test}: image {idx}")

#np.save(os.environ["BRAIN_DATA_DIR"]+'/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub),fmri_test )
#np.save(os.environ["BRAIN_DATA_DIR"]+'/processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub),stim_test )

print("Test data is saved.")
print("All done. Shapes:")
print("fmri_train:", fmri_train.shape)
print("stim_train:", stim_train.shape)

'''annots_cur = np.load(os.environ["BRAIN_DATA_DIR"]+'/annots/COCO_73k_annots_curated.npy',allow_pickle=True)

captions_array = np.empty((num_train,5),dtype=annots_cur.dtype)
for i,idx in enumerate(train_im_idx):
    captions_array[i,:] = annots_cur[idx,:]
    print(f"{i}/{len(train_im_idx)}")
np.save(os.environ["BRAIN_DATA_DIR"]+'/processed_data/subj{:02d}/nsd_train_cap_sub{}.npy'.format(sub,sub),captions_array )
    
captions_array = np.empty((num_test,5),dtype=annots_cur.dtype)
for i,idx in enumerate(test_im_idx):
    captions_array[i,:] = annots_cur[idx,:]
    print(f"{i}/{len(test_im_idx)}")
np.save(os.environ["BRAIN_DATA_DIR"]+'/processed_data/subj{:02d}/nsd_test_cap_sub{}.npy'.format(sub,sub),captions_array )

print("Caption data are saved.")'''
save_path = os.environ["BRAIN_DATA_DIR"]+f"/subj{sub:02d}_fmriflattened_stim_paired.npz"
np.savez_compressed(
    save_path,
    fmri_train=fmri_train,
    stim_train=stim_train,
    fmri_test=fmri_test,
    stim_test=stim_test
)