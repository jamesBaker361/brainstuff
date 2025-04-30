import os

# Download ROIs
for sub in [1,2,5,7]:
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/nsdgeneral.nii.gz nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,sub))
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/floc-faces.nii.gz nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,sub))
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/floc-words.nii.gz nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,sub))
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/floc-places.nii.gz nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,sub))
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/floc-bodies.nii.gz nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub,sub))
