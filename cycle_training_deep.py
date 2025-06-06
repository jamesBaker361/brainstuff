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
parser.add_argument("--fmri_type",type=str,default="voxel",help="array or voxel")
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
        test_loader=DataLoader(test_dataset,batch_size=args.batch_size,)

        pixel_to_fmri=PixelVoxelArrayModel(image_size,fmri_size,args.n_layers,args.n_layers_trans,"pixel",args.fmri_type,args.kernel_size,2,args.residual_blocks)
        fmri_to_pixel=PixelVoxelArrayModel(fmri_size,image_size,args.n_layers,args.n_layers_trans,args.fmri_type,"pixel",args.kernel_size,2,args.residual_blocks)

        #pixel_to_fmri=pixel_to_fmri.to(device)
        #fmri_to_pixel=fmri_to_pixel.to(device)

        trainable_models=[pixel_to_fmri,fmri_to_pixel]

        if args.use_discriminator:
            pixel_discriminator=Discriminator(image_size,args.n_layers_disc,"pixel",args.kernel_size,2,args.residual_blocks)
            fmri_discriminator=Discriminator(fmri_size,args.n_layers_disc,args.fmri_type,args.kernel_size,2,args.residual_blocks)

            

            bce_loss=torch.nn.BCEWithLogitsLoss()
            trainable_models.append(pixel_discriminator)
            trainable_models.append(fmri_discriminator)

        optimizer=torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": 0.0001} for model in trainable_models
            ]
        )

        model=FusedModel(*trainable_models)

        for batch in train_loader:
            break

        if args.unpaired_image_dataset!="":
            model,optimizer,train_loader,test_loader,unpaired_loader=accelerator.prepare(model,optimizer,train_loader,test_loader,unpaired_loader)
        else:
            model,optimizer,train_loader,test_loader=accelerator.prepare(model,optimizer,train_loader,test_loader)
        

        with torch.no_grad():
            gen_img=fmri_to_pixel(fmri.unsqueeze(0))
            print("gen_img max,min,size",gen_img.max(),gen_img.min(),gen_img.size())
            gen_fmri=pixel_to_fmri(img.unsqueeze(0))
            print("gen fmri max,min,size",gen_fmri.max(),gen_fmri.min(),gen_fmri.size())

        def init_loss_dict():
            return {"ptov_reconstruction_loss":[],"vtop_reconstruction_loss":[],"ptov_translation_loss":[],"vtop_translation_loss":[],
                             "fmri_disc_real":[],"fmri_disc_fake":[],"fmri_gen":[],
                             "pixel_disc_real":[],"pixel_disc_fake":[],"pixel_gen":[]}
        
        for e in range(1,args.epochs+1):
            validation_set=[]
            train_loss_dict=init_loss_dict()
            unpaired_train_loss_dict=init_loss_dict()
            val_loss_dict=init_loss_dict()
            start=time.time()
            for k,batch in enumerate(train_loader):
                if k==args.train_limit:
                    break
                with accelerator.accumulate(*trainable_models):
                    if e %args.validation_interval==0 and random.random() < args.val_split:
                        validation_set.append(batch)
                        continue

                    fmri=batch["fmri"].to(device,torch_dtype)
                    images=batch["image"].to(device,torch_dtype)
                    batch_size=images.size()[0]
                    #labels=batch["labels"]

                    if args.use_discriminator:
                        for trainable_model,frozen_model,disc,input_data,output_data,real_key,fake_key,gen_key in [
                            [fmri_to_pixel,pixel_to_fmri,fmri_discriminator,fmri,images,"fmri_disc_real","fmri_disc_fake","fmri_gen"],
                            [pixel_to_fmri,fmri_to_pixel,pixel_discriminator,images,fmri,"pixel_disc_real","pixel_disc_fake","pixel_gen"]]:
                            frozen_model.requires_grad_(False)

                            #train disc real batch
                            disc.requires_grad_(True)
                            trainable_model.requires_grad_(False)
                            optimizer.zero_grad()

                            true_labels=torch.ones((batch_size)).to(device,torch_dtype)
                            '''translated_data=trainable_model(data)
                            reconstructed_data=frozen_model(translated_data)'''
                            predicted_labels=disc(input_data).squeeze(1)
                            d_loss_real=bce_loss(predicted_labels,true_labels)
                            accelerator.backward(d_loss_real)
                            train_loss_dict[real_key].append(d_loss_real.cpu().detach().item())
                            optimizer.step()
                            #torch.cuda.empty_cache()


                            #train disc fake batch
                            optimizer.zero_grad()
                            fake_labels=torch.zeros((batch_size)).to(device,torch_dtype)
                            translated_data=trainable_model(input_data)
                            reconstructed_data=frozen_model(translated_data).detach()
                            predicted_labels=disc(reconstructed_data).squeeze(1)
                            d_loss_fake=bce_loss(predicted_labels,fake_labels)
                            accelerator.backward(d_loss_fake)
                            train_loss_dict[fake_key].append(d_loss_fake.cpu().detach().item())
                            #Sd_loss=d_loss_fake+d_loss_real
                            optimizer.step()

                            #train gen
                            optimizer.zero_grad()
                            disc.requires_grad_(False)
                            trainable_model.requires_grad_(True)
                            true_labels=torch.ones((batch_size)).to(device,torch_dtype)
                            translated_data=trainable_model(input_data)
                            reconstructed_data=frozen_model(translated_data)
                            predicted_labels=disc(reconstructed_data).squeeze(1)
                            gen_loss=bce_loss(predicted_labels,true_labels)
                            accelerator.backward(gen_loss)
                            train_loss_dict[gen_key].append(gen_loss.cpu().detach().item())
                            optimizer.step()

                    if args.reconstruction_loss:

                        for trainable_model,frozen_model,data,key in [
                            [fmri_to_pixel,pixel_to_fmri,fmri,"vtop_reconstruction_loss"],
                            [pixel_to_fmri,fmri_to_pixel,images,"ptov_reconstruction_loss"]]:
                            trainable_model.requires_grad_(True)
                            frozen_model.requires_grad_(False)
                            optimizer.zero_grad()
                            translated_data=trainable_model(data)
                            reconstructed_data=frozen_model(translated_data)
                            loss=F.mse_loss(data,reconstructed_data)
                            train_loss_dict[key].append(loss.cpu().detach().item())
                            accelerator.backward(loss)
                            optimizer.step()
                    if args.translation_loss:
                        for trainable_model,input_data,output_data,key in [
                            [fmri_to_pixel,fmri,images,"vtop_translation_loss"],
                            [pixel_to_fmri,images,fmri,"ptov_translation_loss"]
                        ]:
                            trainable_model.requires_grad_(True)
                            optimizer.zero_grad()
                            translated_data=trainable_model(input_data)
                            loss=F.mse_loss(output_data,translated_data)
                            train_loss_dict[key].append(loss.cpu().detach().item())
                            accelerator.backward(loss)
                            optimizer.step()

            #unpaired training
            if args.unpaired_image_dataset!="":
                for k,images in enumerate(unpaired_loader):
                    if k==args.train_limit:
                        break
                    with accelerator.accumulate(*trainable_models):
                        images=images.to(device,torch_dtype)
                        batch_size=images.size()[0]

                        if args.use_discriminator:
                            for trainable_model,frozen_model,disc,input_data,real_key,fake_key,gen_key in [
                                #[fmri_to_pixel,pixel_to_fmri,optimizer,fmri,fmri_discriminator,optimizer,fmri,images,"fmri_disc_real","fmri_disc_fake","fmri_gen"],
                                [pixel_to_fmri,fmri_to_pixel,pixel_discriminator,images,"pixel_disc_real","pixel_disc_fake","pixel_gen"]]:
                                frozen_model.requires_grad_(False)

                                #train disc real batch
                                disc.requires_grad_(True)
                                trainable_model.requires_grad_(False)
                                optimizer.zero_grad()

                                true_labels=torch.ones((batch_size)).to(device,torch_dtype)
                                '''translated_data=trainable_model(data)
                                reconstructed_data=frozen_model(translated_data)'''
                                predicted_labels=disc(input_data).squeeze(1)
                                d_loss_real=bce_loss(predicted_labels,true_labels)
                                accelerator.backward(d_loss_real)
                                unpaired_train_loss_dict[real_key].append(d_loss_real.cpu().detach().item())
                                optimizer.step()


                                #train disc fake batch
                                optimizer.zero_grad()
                                fake_labels=torch.zeros((batch_size)).to(device,torch_dtype)
                                translated_data=trainable_model(input_data)
                                reconstructed_data=frozen_model(translated_data)
                                predicted_labels=disc(reconstructed_data).squeeze(1)
                                d_loss_fake=bce_loss(predicted_labels,fake_labels)
                                accelerator.backward(d_loss_fake)
                                unpaired_train_loss_dict[fake_key].append(d_loss_fake.cpu().detach().item())
                                #Sd_loss=d_loss_fake+d_loss_real
                                optimizer.step()

                                #train gen
                                optimizer.zero_grad()
                                disc.requires_grad_(False)
                                trainable_model.requires_grad_(True)
                                true_labels=torch.ones((batch_size)).to(device,torch_dtype)
                                translated_data=trainable_model(input_data)
                                reconstructed_data=frozen_model(translated_data)
                                predicted_labels=disc(reconstructed_data).squeeze(1)
                                gen_loss=bce_loss(predicted_labels,true_labels)
                                accelerator.backward(gen_loss)
                                unpaired_train_loss_dict[gen_key].append(gen_loss.cpu().detach().item())
                                optimizer.step()

                        if args.reconstruction_loss:

                            for trainable_model,frozen_model,data,key in [
                                #[fmri_to_pixel,pixel_to_fmri,optimizer,fmri,"vtop_reconstruction_loss"],
                                [pixel_to_fmri,fmri_to_pixel,images,"ptov_reconstruction_loss"]]:
                                trainable_model.requires_grad_(True)
                                frozen_model.requires_grad_(False)
                                optimizer.zero_grad()
                                translated_data=trainable_model(data)
                                reconstructed_data=frozen_model(translated_data)
                                loss=F.mse_loss(data,reconstructed_data)
                                unpaired_train_loss_dict[key].append(loss.cpu().detach().item())
                                accelerator.backward(loss)
                                optimizer.step()
            end=time.time()
            print(f"epoch {e} elapsed {end-start}")
            #validation
            if len(validation_set)!=0:
                with torch.no_grad():
                    for batch in validation_set:
                        
                        fmri=batch["fmri"].to(device,torch_dtype)
                        images=batch["image"].to(device,torch_dtype)
                        batch_size=images.size()[0]
                        #labels=batch["labels"]

                        if args.use_discriminator:
                            for trainable_model,frozen_model,input_data,disc,real_key,fake_key,gen_key in [
                                [fmri_to_pixel,pixel_to_fmri,fmri,fmri_discriminator,"fmri_disc_real","fmri_disc_fake","fmri_gen"],
                                [pixel_to_fmri,fmri_to_pixel,images,pixel_discriminator,"pixel_disc_real","pixel_disc_fake","pixel_gen"]]:
                                frozen_model.requires_grad_(False)
                                trainable_model.requires_grad_(False)
                                disc.requires_grad_(False)

                                true_labels=torch.ones((batch_size)).to(device,torch_dtype)
                                '''translated_data=trainable_model(input_data)
                                reconstructed_data=frozen_model(translated_data)'''
                                predicted_labels=disc(input_data).squeeze(1)
                                d_loss_real=bce_loss(predicted_labels,true_labels)
                                val_loss_dict[real_key].append(d_loss_real.cpu().detach().item())


                                #train disc fake batch
                                fake_labels=torch.zeros((batch_size)).to(device,torch_dtype)
                                translated_data=trainable_model(input_data)
                                reconstructed_data=frozen_model(translated_data)
                                predicted_labels=disc(reconstructed_data).squeeze(1)
                                d_loss_fake=bce_loss(predicted_labels,fake_labels)
                                val_loss_dict[fake_key].append(d_loss_fake.cpu().detach().item())


                                #train gen
                                true_labels=torch.ones((batch_size)).to(device,torch_dtype)
                                translated_data=trainable_model(input_data)
                                reconstructed_data=frozen_model(translated_data)
                                predicted_labels=disc(reconstructed_data).squeeze(1)
                                gen_loss=bce_loss(predicted_labels,true_labels)
                                val_loss_dict[gen_key].append(gen_loss.cpu().detach().item())
                        if args.reconstruction_loss:

                            for trainable_model,frozen_model,data,key in [
                                [fmri_to_pixel,pixel_to_fmri,fmri,"vtop_reconstruction_loss"],
                                [pixel_to_fmri,fmri_to_pixel,images,"ptov_reconstruction_loss"]]:
                                trainable_model.requires_grad_(False)
                                frozen_model.requires_grad_(False)
                                translated_data=trainable_model(data)
                                reconstructed_data=frozen_model(translated_data)
                                loss=F.mse_loss(data,reconstructed_data)
                                val_loss_dict[key].append(loss.cpu().detach().item())

                        if args.translation_loss:
                            for trainable_model,input_data,output_data,key in [
                                [fmri_to_pixel,fmri,images,"vtop_translation_loss"],
                                [pixel_to_fmri,images,fmri,"ptov_translation_loss"]
                            ]:
                                trainable_model.requires_grad_(False)
                                translated_data=trainable_model(input_data)
                                loss=F.mse_loss(output_data,translated_data)
                                val_loss_dict[key].append(loss.cpu().detach().item())
                    for batch in validation_set:
                        fmri=batch["fmri"].to(device,torch_dtype)
                        images=batch["image"].to(device,torch_dtype)
                        break
                    translated_image=fmri_to_pixel(fmri)
                    translated_fmri=pixel_to_fmri(images)
                    reconstructed_image=fmri_to_pixel(translated_fmri)

                    reconstructed_image_list=[]
                    translated_image_list=[]
                    image_list=[]

                    for img_data,data_list in zip([images,translated_image,reconstructed_image],
                                          [image_list,translated_image_list,reconstructed_image_list]):
                        img_np=img_data.cpu().permute(0, 2, 3, 1).float().numpy()
                        img_np=img_np*255
                        img_np=img_np.round().astype(np.uint8)
                        for i in img_np:
                            data_list.append(Image.fromarray(i))

                    for real,translated,reconstructed in zip(image_list,translated_image_list,reconstructed_image_list):
                        concat=concat_images_horizontally(real,translated,reconstructed)
                        accelerator.log({"val_result":wandb.Image(concat)})

            metrics={}
            if e %args.validation_interval==0:
                name_list=["val","train"]
                loss_dict_list=[val_loss_dict,train_loss_dict]
            else:
                name_list=["train"]
                loss_dict_list=[train_loss_dict]
            for name,loss_dict in zip(name_list,loss_dict_list):
                key_list=[]
                if args.use_discriminator:
                    key_list+=["fmri_disc_real","fmri_disc_fake","fmri_gen","pixel_disc_real","pixel_disc_fake","pixel_gen"]
                if args.reconstruction_loss:
                    key_list+=["ptov_reconstruction_loss","vtop_reconstruction_loss"]
                if args.translation_loss:
                    key_list+=["ptov_translation_loss","vtop_translation_loss"]
                for key in key_list:
                    metrics[f"{name}_{key}"]=np.mean(loss_dict[key])

            if args.unpaired_image_dataset!="":
                for name,loss_dict in zip(["unpaired_train"],[unpaired_train_loss_dict]):
                    if args.use_discriminator:
                        key_list=["pixel_disc_real","pixel_disc_fake","pixel_gen"]
                    if args.reconstruction_loss:
                        key_list+=["ptov_reconstruction_loss"]
                    for key in key_list:
                        metrics[f"{name}_{key}"]=np.mean(loss_dict[key])
            
            accelerator.log(metrics)
        reconstructed_image_list=[]
        translated_image_list=[]
        image_list=[]
        with torch.no_grad():
            test_loss_dict={"translation_mse":[],"reconstruction_mse":[]}
            for k,batch in enumerate(test_loader):
                if k==args.test_limit:
                    break
                fmri=batch["fmri"].to(device,torch_dtype)
                images=batch["image"].to(device,torch_dtype)
                batch_size=images.size()[0]
                #labels=batch["labels"]

                translated_image=fmri_to_pixel(fmri)
                #reconstructed_fmri=pixel_to_fmri(translated_image)

                translation_mse=F.mse_loss(translated_image,images).cpu().detach().item()

                test_loss_dict["translation_mse"].append(translation_mse)

                translated_fmri=pixel_to_fmri(images)
                reconstructed_image=fmri_to_pixel(translated_fmri)

                reconstruction_mse=F.mse_loss(reconstructed_image,images).cpu().detach().item()
                test_loss_dict["reconstruction_mse"].append(reconstruction_mse)

                for img_data,data_list in zip([images,translated_image,reconstructed_image],
                                          [image_list,translated_image_list,reconstructed_image_list]):
                    img_np=img_data.cpu().permute(0, 2, 3, 1).float().numpy()
                    img_np=img_np*255
                    img_np=img_np.round().astype(np.uint8)
                    for i in img_np:
                        data_list.append(Image.fromarray(i))
            metrics={}
            for name,loss_dict in zip(["test"],[test_loss_dict]):
                for key in test_loss_dict.keys():
                    metrics[f"{name}_{key}"]=np.mean(loss_dict[key])
            translated_correlation=pixelwise_corr_from_pil(image_list,translated_image_list).mean().item()
            reconstructed_correlation=pixelwise_corr_from_pil(image_list,reconstructed_image_list).mean().item()

            translated_clip=np.mean(clip_difference(image_list,translated_image_list))
            reconstructed_clip=np.mean(clip_difference(image_list,reconstructed_image_list))

            metrics["translated_correlation"]=translated_correlation
            metrics["reconstructed_correlation"]=reconstructed_correlation

            metrics["translated_clip"]=translated_clip
            metrics["reconstructed_clip"]=reconstructed_clip

            print(metrics)

        accelerator.log(metrics)

        for real,translated,reconstructed in zip(image_list,translated_image_list,reconstructed_image_list):
            concat=concat_images_horizontally(real,translated,reconstructed)
            accelerator.log({"test_result":wandb.Image(concat)})

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