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
from modeling import PixelVoxelArrayModel,Discriminator,FusedModel
import random
import torch.nn.functional as F
import wandb
from PIL import Image
import torchvision
from metric_helpers import pixelwise_corr_from_pil,clip_difference



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
parser.add_argument("--kernel_size",type=int,default=4)
parser.add_argument("--n_layers",type=int,default=4)
parser.add_argument("--n_layers_trans",type=int,default=4)
parser.add_argument("--n_layers_disc",type=int,default=4)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--use_discriminator",action="store_true")
parser.add_argument("--sublist",nargs="*",type=int)
parser.add_argument("--fmri_type",type=str,default="array",help="array or voxel")
parser.add_argument("--unpaired_image_dataset",type=str,default="",help="hf path for unpaired images")
parser.add_argument("--key",type=str,default="image",help="image key if using unpaired images")
parser.add_argument("--train_limit",type=int,default=-1,help="limit # of training batches")
parser.add_argument("--test_limit",type=int,help="limit # of testing batches",default=-1)
parser.add_argument("--translation_loss",action="store_true")
parser.add_argument("--reconstruction_loss",action="store_true")
parser.add_argument("--validation_interval",type= int,default=1)
parser.add_argument("--residual_blocks",type=int,default=2)

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
    accelerator=Accelerator(log_with="wandb")
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

        test_fmri=[]
        test_img=[]
        test_labels=[]
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
            fmri_suffix={
                "array":"flattened",
                "voxel":""
            }[args.fmri_type]
            path=os.environ["BRAIN_DATA_DIR"]+f"/subj{sub:02d}_fmri{fmri_suffix}_stim_paired.npz"
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

        train_img=[np.transpose(i, (2,0,1)) for i in train_img]
        test_img=[np.transpose(i, (2,0,1)) for i in test_img]

        if args.fmri_type=="voxel":
            train_fmri=[np.expand_dims(f,0) for f in train_fmri]
            test_fmri=[np.expand_dims(f,0) for f in test_fmri]

        def convert_datatype(x):
            return x.to(torch_dtype)
        '''
        class CenterCropSquare:
            def __call__(self, img):
                # img can be PIL Image or Tensor [C, H, W]
                if isinstance(img, Image.Image):
                    w, h = img.size
                elif torch.is_tensor(img):
                    h, w = img.shape[-2:]
                else:
                    raise TypeError("Unsupported image type")

                size = min(h, w)
                return torchvision.transforms.functional.center_crop(img, output_size=[size, size])

        def normalize(x):
            #assuming x is between 0-255
            return x/255

        transform=torchvision.transforms.Compose([
            CenterCropSquare(),
            torchvision.transforms.Resize((512,512)),
            convert_datatype,
            normalize
        ])

        train_dataset=BrainImageSubjectDataset(train_fmri,train_img,train_labels,args.fmri_type,transform=transform)
        for batch in train_dataset:
            break

        
        img=batch["image"].to(device,torch_dtype)
        fmri=batch["fmri"].to(device,torch_dtype)
        fmri_size=fmri.size()
        image_size=img.size()
        print("img min, max",img.min(),img.max())
        print("fmri min,max",fmri.min(),fmri.max())

        print("fmri size",fmri_size)
        print("image size",image_size)

        

        test_dataset=BrainImageSubjectDataset(test_fmri,test_img,test_labels,args.fmri_type,transform=transform)

        if args.unpaired_image_dataset!="":
            try:
                unpaired_dataset=UnpairedImageDataset(args.unpaired_image_dataset,args.key,transform=transform)
            except OSError:
                time.sleep(5+20*random.random())
                try:
                    unpaired_dataset=UnpairedImageDataset(args.unpaired_image_dataset,args.key,transform=transform)
                except OSError:
                    unpaired_dataset=UnpairedImageDataset(args.unpaired_image_dataset,args.key,transform=transform,force_download=True)
            unpaired_loader=DataLoader(unpaired_dataset,batch_size=args.batch_size,shuffle=True)

        train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
        test_loader=DataLoader(test_dataset,batch_size=args.batch_size,)'''

        from sklearn.decomposition import PCA

        pca = PCA(n_components=0.95)
        pca.fit(train_fmri)
        important_features = np.argsort(np.abs(pca.components_[0]))

        print("len",len(important_features))
        print(important_features[-10:])


        


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