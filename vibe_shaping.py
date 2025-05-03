import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
args = parser.parse_args()
sub = int(args.sub)
assert sub in [1, 2, 5, 7]

# Load .mat file helper
def loadmat(filename):
    def _check_keys(d):
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
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

# Load stimulus ordering
stim_order_f = os.environ["BRAIN_DATA_DIR"] + '/nsddata/experiments/nsd/nsd_expdesign.mat'
stim_order = loadmat(stim_order_f)

# Get trial -> image mapping
sig_train = {}
sig_test = {}
num_trials = 37 * 750  # 27750

for idx in range(num_trials):
    nsdId = stim_order['subjectim'][sub - 1, stim_order['masterordering'][idx] - 1] - 1
    if stim_order['masterordering'][idx] > 1000:
        sig_train.setdefault(nsdId, []).append(idx)
    else:
        sig_test.setdefault(nsdId, []).append(idx)

train_im_idx = list(sig_train.keys())
test_im_idx = list(sig_test.keys())

# Load all fMRI volumes (no masking)
betas_dir = os.environ["BRAIN_DATA_DIR"] + f'/nsddata_betas/ppdata/subj{sub:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'
example_beta = nib.load(betas_dir + "betas_session01.nii.gz")
vol_shape = example_beta.shape[:3]
fmri_4d = np.zeros(vol_shape + (num_trials,), dtype=np.float32)

for i in range(37):
    beta_f = nib.load(betas_dir + f"betas_session{i+1:02d}.nii.gz").get_fdata().astype(np.float32)
    fmri_4d[..., i * 750:(i + 1) * 750] = beta_f
    if i <3:
        print('beta_f.shape',beta_f.shape)
    del beta_f

print("fMRI data loaded with shape:", fmri_4d.shape)

# Load stimuli
f_stim = h5py.File(os.environ["BRAIN_DATA_DIR"] + '/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
stim = f_stim['imgBrick'][:]
print("Stimuli loaded:", stim.shape)

# Assemble training fMRI/stim data
num_train = len(train_im_idx)
im_dim, im_c = 425, 3
fmri_train = np.zeros((num_train,) + vol_shape, dtype=np.float32)
stim_train = np.zeros((num_train, im_dim, im_dim, im_c), dtype=np.uint8)

for i, idx in enumerate(train_im_idx):
    stim_train[i] = stim[idx]
    trial_indices = sorted(sig_train[idx])  # all trials for this image
    fmri_train[i] = np.mean(fmri_4d[..., trial_indices], axis=-1)  # mean over trials
    if i <3:
        print('fmri_train[i].shape',fmri_train[i].shape)
        print('sorted(sig_train[idx])',sorted(sig_train[idx]))
        print('trial_indices',trial_indices)
        print(f"Train {i+1}/{num_train}: image {idx} from {len(trial_indices)} trials")

# Assemble test fMRI/stim data similarly (optional)
num_test = len(test_im_idx)
fmri_test = np.zeros((num_test,) + vol_shape, dtype=np.float32)
stim_test = np.zeros((num_test, im_dim, im_dim, im_c), dtype=np.uint8)
for i, idx in enumerate(test_im_idx):
    stim_test[i] = stim[idx]
    trial_indices = sorted(sig_test[idx])
    fmri_test[i] = np.mean(fmri_4d[..., trial_indices], axis=-1)
    if i <3:
        print('fmri_test[i].shape',fmri_test[i].shape)
        print('sorted(sig_test[idx])',sorted(sig_test[idx]))
        print('trial_indices',trial_indices)
        print(f"Test {i+1}/{num_test}: image {idx} from {len(trial_indices)} trials")

print("All done. Shapes:")
print("fmri_train:", fmri_train.shape)
print("stim_train:", stim_train.shape)

# Save paired data to .npz
save_path = os.environ["BRAIN_DATA_DIR"]+f"/subj{sub:02d}_fmri_stim_paired.npz"
np.savez_compressed(
    save_path,
    fmri_train=fmri_train,
    stim_train=stim_train,
    fmri_test=fmri_test,
    stim_test=stim_test
)

print(f"Saved paired data to {save_path}")
