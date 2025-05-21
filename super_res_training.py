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
from pathlib import Path
import re
from datasets import Dataset

def get_max_file(save_dir,name):
    directory = Path(save_dir)

    # Regular expression to match files like name_123.pth
    pattern = re.compile(rf"^{re.escape(name)}_(\d+)\.pth$")

    max_e = -1
    max_file = None

    for file in directory.glob(f"{name}_*.pth"):
        match = pattern.match(file.name)
        if match:
            e = int(match.group(1))
            if e > max_e:
                max_e = e
                max_file = file
    return max_file,max_e


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
parser.add_argument("--pretest_limit",type=int,default=10)
parser.add_argument("--name",type=str,default="")
parser.add_argument("--save_interval",type=int,default=50)
parser.add_argument("--load",action="store_true")

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
    if args.deepspeed==True:
        accelerator=Accelerator(log_with="wandb")
        print("deepspeed training!")
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

    save_dir=os.path.join(os.environ["BRAIN_DATA_DIR"], "models")
    os.makedirs(save_dir, exist_ok=True)
    with accelerator.autocast():

        print("torch dtype",torch_dtype)

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
            fmri_suffix="flattened"
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

        n_val_indices=int(args.val_split*len(train_fmri))
        val_indices=set(np.random.choice(np.arange(0, len(train_fmri)), size=n_val_indices, replace=False))
        print("n_val_indices",n_val_indices)
        train_img=[np.transpose(i, (2,0,1)) for i in train_img]

        test_img=[np.transpose(i, (2,0,1)) for i in test_img]

        pca = joblib.load(os.path.join(os.environ["BRAIN_DATA_DIR"],"pca",f"subj{sub:02d}_4096_fmri.pkl"))

        train_fmri=pca.transform(train_fmri)
        test_fmri=pca.transform(test_fmri)

        print('train_fmri.shape,test_fmri.shape',train_fmri.shape,test_fmri.shape)

        train_fmri=[np.array(row).reshape(256,4,4) for row in train_fmri]
        test_fmri=[np.array(row).reshape(256,4,4) for row in test_fmri]

        def convert_datatype(x):
            return x.to(torch_dtype)
        
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

        train_dataset=BrainImageSubjectDataset(train_fmri,train_img,train_labels,"array",transform=transform)
        for batch in train_dataset:
            break

        
        img=batch["image"].to(device).to(torch_dtype)
        fmri=batch["fmri"].to(device).to(torch_dtype)
        fmri_size=fmri.size()
        image_size=img.size()
        print("img min, max",img.min(),img.max())
        print("fmri min,max",fmri.min(),fmri.max())

        print("fmri size",fmri_size)
        print("image size",image_size)

        

        test_dataset=BrainImageSubjectDataset(test_fmri,test_img,test_labels,"array",transform=transform)

        train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
        test_loader=DataLoader(test_dataset,batch_size=args.batch_size,)

        model=SuperResolutionModel((256,4,4),(3,512,512),args.residual_blocks)
        start_epoch=1

        max_file,max_e=get_max_file(save_dir,args.name)
        if max_file is not None and args.load:
            print("loading from ",max_file)
            model.load_state_dict(torch.load(max_file,weights_only=True))
            start_epoch=max_e
        model=model.to(device).to(torch_dtype)

        # If using torch_dtype=torch.float16, also convert manually:
        '''if torch_dtype == torch.float16:
            for p in model.parameters():
                p.data = p.data.half()'''

        optimizer=torch.optim.AdamW([p for p in model.parameters()],0.001)
        try:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataset)*args.epochs,verbose=False)
        except TypeError:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataset)*args.epochs)

        
        
        model,optimizer,scheduler,train_loader,test_loader=accelerator.prepare(model,optimizer,scheduler,train_loader,test_loader)

        #PREtesting
        reconstructed_image_list=[]
        image_list=[]
        metrics={}

        hf_dataset_dict={
            "src":[],
            "before":[],
            "after":[]
        }

        with torch.no_grad():
            test_loss_list=[]
            for k,batch in enumerate(test_loader):
                if k==args.test_limit:
                    break
                fmri=batch["fmri"].to(device).to(torch_dtype)
                images=batch["image"].to(device).to(torch_dtype)
                batch_size=images.size()[0]

                reconstructed_images=model(fmri)
                loss=F.mse_loss(images,reconstructed_images)
                test_loss_list.append(loss.cpu().detach().item())

                for img_data,data_list in zip([images,reconstructed_images],
                                        [image_list,reconstructed_image_list]):
                    img_np=img_data.cpu().permute(0, 2, 3, 1).float().numpy()
                    img_np=img_np*255
                    img_np=img_np.round().astype(np.uint8)
                    for i in img_np:
                        data_list.append(Image.fromarray(i))

            

            reconstructed_clip=np.mean(clip_difference(image_list,reconstructed_image_list))

            metrics={
                "test_loss":np.mean(test_loss_list),
                "clip_difference":reconstructed_clip
            }
            print(metrics)
            for k,(real,reconstructed) in enumerate(zip(image_list,reconstructed_image_list)):
                #concat=concat_images_horizontally(real,reconstructed)
                hf_dataset_dict["src"].append(real)
                hf_dataset_dict["before"].append(reconstructed)
                #metrics[f"test_result_{k}"]=wandb.Image(concat)
            accelerator.log(metrics)

        for e in range(start_epoch, args.epochs+1):
            start=time.time()
            validation_set=[]
            train_loss_list=[]
            for k,batch in enumerate(train_loader):
                if k==args.train_limit:
                    break
                with accelerator.accumulate(model):
                    if k in val_indices:
                        validation_set.append(batch)
                        continue
                    fmri=batch["fmri"].to(device).to(torch_dtype)
                    images=batch["image"].to(device).to(torch_dtype)
                    batch_size=images.size()[0]

                    optimizer.zero_grad()
                    reconstructed_images=model(fmri)
                    loss=F.mse_loss(images,reconstructed_images)
                    train_loss_list.append(loss.cpu().detach().item())
                    accelerator.backward(loss)
                    optimizer.step()
                scheduler.step()
            if e % args.save_interval==0:
                path=os.path.join(save_dir, f"{args.name}_{e}.pth")
                torch.save(model.state_dict(),path)
            metrics={
                "training_loss":np.mean(train_loss_list)
            }
            end=time.time()
            print(f"epoch {e} elapsed {end-start}")
            #validation
            if e%args.validation_interval==0:
                val_loss_list=[]
                with torch.no_grad():
                    for batch in validation_set:
                        
                        fmri=batch["fmri"].to(device).to(torch_dtype)
                        images=batch["image"].to(device).to(torch_dtype)
                        batch_size=images.size()[0]

                        reconstructed_images=model(fmri)
                        loss=F.mse_loss(images,reconstructed_images)
                        val_loss_list.append(loss.cpu().detach().item())
                    metrics["val_loss"]=np.mean(val_loss_list)
                    for batch in validation_set:
                        fmri=batch["fmri"].to(device).to(torch_dtype)
                        images=batch["image"].to(device).to(torch_dtype)
                        break
                    reconstructed_images=model(fmri)

                    reconstructed_image_list=[]
                    image_list=[]

                    for img_data,data_list in zip([images,reconstructed_images],
                                            [image_list,reconstructed_image_list]):
                        img_np=img_data.cpu().permute(0, 2, 3, 1).float().numpy()
                        img_np=img_np*255
                        img_np=img_np.round().astype(np.uint8)
                        for i in img_np:
                            data_list.append(Image.fromarray(i))

                    for k,(real,reconstructed) in enumerate(zip(image_list,reconstructed_image_list)):
                        concat=concat_images_horizontally(real,reconstructed)
                        #accelerator.log({"val_result":wandb.Image(concat)})
                        metrics[f"val_result_{k}"]=wandb.Image(concat)
            accelerator.log(metrics)
        
        path=os.path.join(save_dir, f"{args.name}_{e}.pth")
        torch.save(model.state_dict(),path)
        #testing
        reconstructed_image_list=[]
        image_list=[]
        metrics={}
        with torch.no_grad():
            test_loss_list=[]
            for k,batch in enumerate(test_loader):
                if k==args.test_limit:
                    break
                fmri=batch["fmri"].to(device).to(torch_dtype)
                images=batch["image"].to(device).to(torch_dtype)
                batch_size=images.size()[0]

                reconstructed_images=model(fmri)
                loss=F.mse_loss(images,reconstructed_images)
                test_loss_list.append(loss.cpu().detach().item())

                for img_data,data_list in zip([images,reconstructed_images],
                                        [image_list,reconstructed_image_list]):
                    img_np=img_data.cpu().permute(0, 2, 3, 1).float().numpy()
                    img_np=img_np*255
                    img_np=img_np.round().astype(np.uint8)
                    for i in img_np:
                        data_list.append(Image.fromarray(i))

            reconstructed_clip=np.mean(clip_difference(image_list,reconstructed_image_list))

            metrics={
                "test_loss":np.mean(test_loss_list),
                "clip_difference":reconstructed_clip
            }
            print(metrics)
            for k,(real,reconstructed) in enumerate(zip(image_list,reconstructed_image_list)):
                #concat=concat_images_horizontally(real,reconstructed)
                #metrics[f"test_result_{k}"]=wandb.Image(concat)
                hf_dataset_dict["after"].append(reconstructed)
            accelerator.log(metrics)
            try:
                Dataset.from_dict(hf_dataset_dict).push_to_hub(f"jlbaker361/{args.name}")
            except:
                with open("token.txt","r") as file:
                    token=file.readline().strip()
                    Dataset.from_dict(hf_dataset_dict).push_to_hub(f"jlbaker361/{args.name}",token=token)
                


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