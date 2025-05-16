import torch
import torchvision.transforms.functional
from gpu_helpers import *
from accelerate import Accelerator
import argparse
import time
import numpy as np
import os
from torch.utils.data import DataLoader
from data_helpers import BrainImageSubjectDataset,UnpairedImageDataset
from modeling import PixelVoxelArrayModel,Discriminator,FusedModel,SuperResolutionModel
import random
import torch.nn.functional as F
import wandb
from PIL import Image
import torchvision
from metric_helpers import pixelwise_corr_from_pil,clip_difference
from sklearn.decomposition import PCA
import joblib


for i in range(torch.cuda.device_count()):
    try:
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.memory_allocated(i)
    except RuntimeError as e:
        print(f"[WARN] GPU {i} has a problem: {e}")

torch.cuda.empty_cache()

parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--batch_size",type=int,default=4)
parser.add_argument("--val_split",type=float,default=0.1)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--sublist",nargs="*",type=int)
parser.add_argument("--train_limit",type=int,default=-1,help="limit # of training batches")
parser.add_argument("--test_limit",type=int,help="limit # of testing batches",default=-1)
parser.add_argument("--validation_interval",type= int,default=1)
parser.add_argument("--residual_blocks",type=int,default=2)
parser.add_argument("--deepspeed",action="store_true")

def concat_images_horizontally(*imgs: Image.Image) -> Image.Image:
    """
    Concatenate a variable number of PIL images horizontally.
    All images must have the same height.
    """
    if len(imgs) == 0:
        raise ValueError("At least one image must be provided.")

    # Convert all images to RGB (optional, but helpful)
    imgs = [img.convert("RGB") for img in imgs]

    # Check that all heights match
    height = imgs[0].height
    if not all(img.height == height for img in imgs):
        raise ValueError("All images must have the same height for horizontal concatenation.")

    total_width = sum(img.width for img in imgs)
    new_img = Image.new('RGB', (total_width, height))

    x_offset = 0
    for img in imgs:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img

def main(args):
    if args.deepspeed:
        accelerator=Accelerator(log_with="wandb")
    else:
        accelerator=Accelerator(log_with="wandb",
                                mixed_precision=args.mixed_precision,
                                gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    device=accelerator.device
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    with accelerator.autocast():

        train_fmri=[]
        train_img=[]
        train_labels=[]
        train_text=[]

        test_fmri=[]
        test_img=[]
        test_labels=[]
        test_text=[]
        
        subject_class_labels={
            1:np.array([1,0,0,0]),
            2:np.array([0,1,0,0]),
            5:np.array([0,0,1,0]),
            7:np.array([0,0,0,1])
        }
        if args.sublist==None:
            sublist=[1, 2, 5, 7]
        else:
            sublist=args.sublist
        for sub in sublist:
            fmri_suffix="flattened"
            path=os.environ["BRAIN_DATA_DIR"]+f"/subj{sub:02d}_fmri{fmri_suffix}_stim_paired.npz"
            npz_loaded=np.load(path,allow_pickle=True)
            
            subject_captions_test=npz_loaded["captions_test"]
            subject_captions_train=npz_loaded["captions_train"]

            print(subject_captions_test[0])
            print(subject_captions_train[0])


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")