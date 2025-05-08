import torch
from gpu_helpers import *
from accelerate import Accelerator
import argparse
import time
import numpy as np
import os
from torch.utils.data import DataLoader
from data_helpers import BrainImageSubjectDataset
from modeling import PixelVoxelArrayModel,Discriminator
import random
import torch.nn.functional as F
import wandb
from PIL import Image
import torchvision

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
parser.add_argument("--fmri_type",type=str,default="voxel",help="array or voxel")
parser.add_argument("--unpaired_image_dataset",type=str,default="",help="hf path for unpaired images")
parser.add_argument("--key",type=str,default="image",help="image key if using unpaired images")

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
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
            subject_stim_test=npz_loaded["fmri_test"]

            n_test=len(subject_fmri_test)
            n_train=len(subject_fmri_train)

            subject_train_labels=[subject_class_labels[sub] for _ in range(n_train)]
            subject_test_labels=[subject_class_labels[sub] for _ in range(n_test)]

            train_fmri.extend(subject_fmri_train)
            print(sub,'train_fmri.max(),train_fmri.min()',train_fmri.max(),train_fmri.min())
            train_img.extend(subject_stim_train)
            train_labels.extend(subject_train_labels)

            test_fmri.extend(subject_fmri_test)
            print(sub,'test_fmri.max(),test_fmri.min()',test_fmri.max(),test_fmri.min())
            test_img.extend(subject_stim_test)
            test_labels.extend(subject_test_labels)

        train_img=[np.transpose(i, (2,0,1)) for i in train_img]
        test_img=[np.transpose(i, (2,0,1)) for i in test_img]

        if args.fmri_type=="voxel":
            train_fmri=[np.expand_dims(f,0) for f in train_fmri]
            test_fmri=[np.expand_dims(f,0) for f in test_fmri]

        def convert_datatype(x):
            return x.to(torch_dtype)

        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((512,512)),
            convert_datatype,
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_dataset=BrainImageSubjectDataset(train_fmri,train_img,train_labels,transform=transform)
        for batch in train_dataset:
            break

        fmri_size=batch["fmri"].size()
        image_size=batch["image"].size()

        print("fmri size",fmri_size)
        print("image size",image_size)

        

        test_dataset=BrainImageSubjectDataset(test_fmri,test_img,test_labels,transform=transform)

        train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
        test_loader=DataLoader(test_dataset,batch_size=args.batch_size,)

        pixel_to_voxel=PixelVoxelArrayModel(image_size,fmri_size,args.n_layers,args.n_layers_trans,"pixel",args.fmri_type,args.kernel_size,2)
        voxel_to_pixel=PixelVoxelArrayModel(fmri_size,image_size,args.n_layers,args.n_layers_trans,args.fmri_type,"pixel",args.kernel_size,2)

        ptov_optimizer=torch.optim.AdamW([p for p in pixel_to_voxel.parameters()])
        vtop_optimizer=torch.optim.AdamW([p for p in voxel_to_pixel.parameters()])

        pixel_to_voxel,voxel_to_pixel,ptov_optimizer,vtop_optimizer=accelerator.prepare(pixel_to_voxel,voxel_to_pixel,ptov_optimizer,vtop_optimizer)

        if args.use_discriminator:
            pixel_discriminator=Discriminator(image_size,args.n_layer,"pixel",args.kernel_size)
            voxel_discriminator=Discriminator(fmri_size,args.n_layers,args.fmri_type,args.kernel_size)

            pdisc_optimizer=torch.optim.AdamW([p for p in pixel_discriminator.parameters()])
            vdisc_optimizer=torch.optim.AdamW([p for p in voxel_discriminator.parameters()])

            pixel_discriminator,voxel_discriminator,pdisc_optimizer,vdisc_optimizer=accelerator.prepare(pixel_discriminator,voxel_discriminator,pdisc_optimizer,vdisc_optimizer)
            bce_loss=torch.nn.BCEWithLogitsLoss()
        for batch in train_loader:
            break

        img=batch["image"].to(device,torch_dtype)
        fmri=batch["fmri"].to(device,torch_dtype)
        print("img min, max",img.min(),img.max())
        print("fmri min,max",fmri.min(),fmri.max())

        with torch.no_grad():
            gen_img=voxel_to_pixel(fmri)
            print("gen_img max,min,size",gen_img.max(),gen_img.min(),gen_img.size())
            gen_fmri=pixel_to_voxel(img)
            print("gen fmri max,min,size",gen_fmri.max(),gen_fmri.min(),gen_fmri.size())
        img=img.unsqueeze(0).cpu().permute(0, 2, 3, 1).float().numpy()
        try:
            pil_img=Image.fromarray(img) #good as is
            pil_img.save("img.png")
            accelerator.log({
                "pil_img":wandb.Image(pil_img),
            })
        except Exception as e:
            print("pil_img=Image.fromarray(img) failed")
            print(e)

        try:
            pil_rescaled_img=Image.fromarray(img*255) #assuming its [0,1]
            pil_rescaled_img.save("img_rescaled.png")
            accelerator.log({
                "pil_rescaled_img":wandb.Image(pil_rescaled_img)
            })
        except Exception as e:
            print("Image.fromarray(img*255) failed")
            print(e)

        
        try:
            pil_rescaled_shifted_img=Image.fromarray(img*255 +128) #assuming its [-1,1]
            pil_rescaled_img.save("img_rescaled_shifted.png")
            accelerator.log({
                "pil_rescaled_shifted_img":wandb.Image(pil_rescaled_shifted_img)
            })
        except Exception as e:
            print("Image.fromarray(img*255 +128) failed")
            print(e)



        def init_loss_dict():
            return {"ptov_loss":[],"vtop_loss":[],
                             "voxel_disc_real":[],"voxel_disc_fake":[],"voxel_gen":[],
                             "pixel_disc_real":[],"pixel_disc_fake":[],"pixel_gen":[]}
        
        for e in range(1,args.epochs+1):
            validation_set=[]
            train_loss_dict=init_loss_dict()
            val_loss_dict=init_loss_dict()
            for batch in train_loader:
                with accelerator.accumulate():
                    if random.random() < args.val_split:
                        validation_set.append(batch)
                        continue

                    fmri=batch["fmri"].to(device,torch_dtype)
                    images=batch["image"].to(device,torch_dtype)
                    labels=batch["labels"]

                    if args.use_discriminator:
                        for trainable_model,frozen_model,gen_optimizer,disc,disc_optimizer,real_key,fake_key,gen_key in zip([
                            [voxel_to_pixel,pixel_to_voxel,vtop_optimizer,fmri,voxel_discriminator,vdisc_optimizer,"voxel_disc_real","voxel_disc_fake","voxel_gen"],
                            [pixel_to_voxel,voxel_to_pixel,ptov_optimizer,images,pixel_discriminator,pdisc_optimizer,"pixel_disc_real","pixel_disc_fake","pixel_gen"]]):
                            frozen_model.requires_grad_(False)

                            #train disc real batch
                            disc.requires_grad_(True)
                            trainable_model.requires_grad_(False)
                            disc_optimizer.zero_grad()

                            true_labels=torch.ones((args.batch_size))
                            translated_data=trainable_model(data)
                            reconstructed_data=frozen_model(translated_data)
                            predicted_labels=disc(reconstructed_data)
                            d_loss_real=bce_loss(predicted_labels,true_labels)
                            accelerator.backward(d_loss_real)
                            train_loss_dict[real_key].append(d_loss_real.cpu().detach().item())


                            #train disc fake batch
                            fake_labels=torch.zeros((args.batch_size))
                            translated_data=trainable_model(data)
                            reconstructed_data=frozen_model(translated_data)
                            predicted_labels=disc(reconstructed_data)
                            d_loss_fake=bce_loss(predicted_labels,fake_labels)
                            accelerator.backward(d_loss_fake)
                            train_loss_dict[fake_key].append(d_loss_fake.cpu().detach().item())
                            #Sd_loss=d_loss_fake+d_loss_real
                            disc_optimizer.step()

                            #train gen
                            gen_optimizer.zero_grad()
                            disc.requires_grad_(False)
                            trainable_model.requires_grad_(True)
                            true_labels=torch.ones((args.batch_size))
                            translated_data=trainable_model(data)
                            reconstructed_data=frozen_model(translated_data)
                            predicted_labels=disc(reconstructed_data)
                            gen_loss=bce_loss(predicted_labels,true_labels)
                            accelerator.backward(gen_loss)
                            train_loss_dict[gen_key].append(gen_loss.cpu().detach().item())
                            gen_optimizer.step()

                    else:

                        for trainable_model,frozen_model,optimizer,data,key in zip([
                            [voxel_to_pixel,pixel_to_voxel,vtop_optimizer,fmri,"vtop_loss"],
                            [pixel_to_voxel,voxel_to_pixel,ptov_optimizer,images,"ptov_loss"]]):
                            trainable_model.requires_grad_(True)
                            frozen_model.requires_grad_(False)
                            optimizer.zero_grad()
                            translated_data=trainable_model(data)
                            reconstructed_data=frozen_model(translated_data)
                            loss=F.mse_loss(data,reconstructed_data)
                            train_loss_dict[key].append(loss.cpu().detach().item())
                            accelerator.backward(loss)
                            optimizer.step()
            with torch.no_grad():
                for batch in validation_set:
                    fmri=batch["fmri"].to(device,torch_dtype)
                    images=batch["image"].to(device,torch_dtype)
                    labels=batch["labels"]

                    if args.use_discriminator:
                        for trainable_model,frozen_model,gen_optimizer,disc,disc_optimizer,real_key,fake_key,gen_key in zip([
                            [voxel_to_pixel,pixel_to_voxel,vtop_optimizer,fmri,voxel_discriminator,vdisc_optimizer,"voxel_disc_real","voxel_disc_fake","voxel_gen"],
                            [pixel_to_voxel,voxel_to_pixel,ptov_optimizer,images,pixel_discriminator,pdisc_optimizer,"pixel_disc_real","pixel_disc_fake","pixel_gen"]]):
                            frozen_model.requires_grad_(False)
                            trainable_model.requires_grad_(False)
                            disc.requires_grad_(False)

                            true_labels=torch.ones((args.batch_size))
                            translated_data=trainable_model(data)
                            reconstructed_data=frozen_model(translated_data)
                            predicted_labels=disc(reconstructed_data)
                            d_loss_real=bce_loss(predicted_labels,true_labels)
                            val_loss_dict[real_key].append(d_loss_real.cpu().detach().item())


                            #train disc fake batch
                            fake_labels=torch.zeros((args.batch_size))
                            translated_data=trainable_model(data)
                            reconstructed_data=frozen_model(translated_data)
                            predicted_labels=disc(reconstructed_data)
                            d_loss_fake=bce_loss(predicted_labels,fake_labels)
                            val_loss_dict[fake_key].append(d_loss_fake.cpu().detach().item())


                            #train gen
                            disc.requires_grad_(False)
                            trainable_model.requires_grad_(True)
                            true_labels=torch.ones((args.batch_size))
                            translated_data=trainable_model(data)
                            reconstructed_data=frozen_model(translated_data)
                            predicted_labels=disc(reconstructed_data)
                            gen_loss=bce_loss(predicted_labels,true_labels)
                            val_loss_dict[gen_key].append(gen_loss.cpu().detach().item())
                    else:

                        for trainable_model,frozen_model,optimizer,data,key in zip([
                            [voxel_to_pixel,pixel_to_voxel,vtop_optimizer,fmri,"vtop_loss"],
                            [pixel_to_voxel,voxel_to_pixel,ptov_optimizer,images,"ptov_loss"]]):
                            trainable_model.requires_grad_(False)
                            frozen_model.requires_grad_(False)
                            translated_data=trainable_model(data)
                            reconstructed_data=frozen_model(translated_data)
                            loss=F.mse_loss(data,reconstructed_data)
                            val_loss_dict[key].append(loss.cpu().detach().item())
            metrics={}
            for name,loss_dict in zip(["val","train"],[val_loss_dict,train_loss_dict]):
                if args.use_discriminator:
                    key_list=["voxel_disc_real","voxel_disc_fake","voxel_gen","pixel_disc_real","pixel_disc_fake","pixel_gen"]
                else:
                    key_list=["ptov_loss","vtop_loss"]
                for key in key_list:
                    metrics[f"{name}_{key}"]=np.mean(loss_dict[key])
            
            accelerator.log(metrics)
        reconstructed_image_list=[]
        with torch.no_grad():
            test_loss_dict=init_loss_dict()
            for batch in test_loader:
                fmri=batch["fmri"].to(device,torch_dtype)
                images=batch["image"].to(device,torch_dtype)
                labels=batch["labels"]

                for trainable_model,frozen_model,gen_optimizer,disc,disc_optimizer,real_key,fake_key,gen_key in zip([
                            [voxel_to_pixel,pixel_to_voxel,vtop_optimizer,fmri,voxel_discriminator,vdisc_optimizer,"voxel_disc_real","voxel_disc_fake","voxel_gen"],
                            #[pixel_to_voxel,voxel_to_pixel,ptov_optimizer,images,pixel_discriminator,pdisc_optimizer,"pixel_disc_real","pixel_disc_fake","pixel_gen"]
                            ]):
                    if args.use_discriminator:
                        true_labels=torch.ones((args.batch_size))
                        translated_data=trainable_model(data)
                        reconstructed_data=frozen_model(translated_data)
                        predicted_labels=disc(reconstructed_data)
                        gen_loss=bce_loss(predicted_labels,true_labels)
                        test_loss_dict[gen_key].append(gen_loss.cpu().detach().item())

                    else:
                        for trainable_model,frozen_model,optimizer,data,key in zip([
                            [voxel_to_pixel,pixel_to_voxel,vtop_optimizer,fmri,"vtop_loss"],
                            [pixel_to_voxel,voxel_to_pixel,ptov_optimizer,images,"ptov_loss"]]):
                            trainable_model.requires_grad_(False)
                            frozen_model.requires_grad_(False)
                            translated_data=trainable_model(data)
                            reconstructed_data=frozen_model(translated_data)
                            loss=F.mse_loss(data,reconstructed_data)
                            test_loss_dict[key].append(loss.cpu().detach().item())
                    reconstructed_image_list.append(reconstructed_data)
            metrics={}
            for name,loss_dict in zip(["test"],[test_loss_dict]):
                if args.use_discriminator:
                    key_list=["voxel_gen","pixel_gen"]
                else:
                    key_list=["ptov_loss","vtop_loss"]
                for key in key_list:
                    metrics[f"{name}_{key}"]=np.mean(loss_dict[key])
        accelerator.log(metrics)
        #log images and maybe score their realism???





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