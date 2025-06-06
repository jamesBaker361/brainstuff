import os
#os.system('ls -l')

# Download Experiment Infos
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat '+os.environ["BRAIN_DATA_DIR"]+'/nsddata/experiments/nsd/')
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl  ' +os.environ["BRAIN_DATA_DIR"]+'/nsddata/experiments/nsd/')

# Download Stimuli
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 '+os.environ["BRAIN_DATA_DIR"]+'/nsddata_stimuli/stimuli/nsd/')

# Download Betas
for sub in [1,2,5,7]:
    for sess in range(1,38):
        os.system('aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{:02d}.nii.gz {}/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub,sess,os.environ["BRAIN_DATA_DIR"],sub))

# Download ROIs
for sub in [1,2,5,7]:
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/nsdgeneral.nii.gz {}/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,os.environ["BRAIN_DATA_DIR"],sub))
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/floc-faces.nii.gz {}/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,os.environ["BRAIN_DATA_DIR"],sub))
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/floc-words.nii.gz {}/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,os.environ["BRAIN_DATA_DIR"],sub))
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/floc-places.nii.gz {}/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,os.environ["BRAIN_DATA_DIR"],sub))
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/floc-bodies.nii.gz {}/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,os.environ["BRAIN_DATA_DIR"],sub))
