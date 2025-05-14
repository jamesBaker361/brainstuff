import torch
from torch import nn
from functools import reduce
import operator
from deep_modeling import ArrayBlock, PixelBlock

def compute_input_size_1d(output_size, n_layers, factor):
    dim=output_size[-1]
    for _ in range(n_layers):
        dim=dim//factor
    return [dim]

def compute_input_size_2d(output_size, n_layers, factor):
    dim=output_size[-1]
    for _ in range(n_layers):
        dim=dim//factor
    return dim,dim


def compute_input_size_3d(output_size, n_layers,
                          factor):
    dim=output_size[-1]
    for _ in range(n_layers):
        dim=dim//factor
    return dim,dim,dim

class ReshapeLayer(nn.Module):
    def __init__(self,dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim=dim

    def forward(self,x):
        batch_size=x.size()[0]
        return x.reshape(batch_size, *self.dim)
class PixelVoxelArrayModel(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_layers,
                 n_layers_trans:int,
                  input_modality:str, #one of voxel or pixel or array
                  output_modality:str, #one of voxel or pixel or array
                  kernel_size:int,
                  factor:int,
                 residual_blocks:int,
                    *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.residual_blocks=residual_blocks

        
        
        stride=kernel_size//factor
        padding=stride//factor
        layers=[]
        in_channels=input_dim[0]
        size_function={
            "voxel":compute_input_size_3d,
            "pixel":compute_input_size_2d,
            "array":compute_input_size_1d
        }[output_modality]
        down_layer={
            "voxel":nn.Conv3d,
            "pixel":nn.Conv2d,
            "array":nn.Linear
        }[input_modality]
        norm={
            "voxel":nn.BatchNorm3d,
            "pixel":nn.BatchNorm2d,
            "array":nn.LayerNorm
        }[input_modality]
        up_layer={
            "voxel":nn.ConvTranspose3d,
            "pixel":nn.ConvTranspose2d,
            "array":nn.Linear
        }[output_modality]
        down_layer_list=[]
        shape=input_dim
        zero_input=torch.zeros(input_dim).unsqueeze(0)
        if input_modality=="voxel" or input_modality=="pixel":
            down_layer_list.append(down_layer(shape[0],4,1,1))
            
            shape=(4, *shape[1:])
        target_shape=shape
        print('shape',shape)
        for _ in range(n_layers):
            in_channels=shape[0]
            if input_modality=="array":
                for _ in range(residual_blocks):
                  down_layer_list.append(ArrayBlock(in_channels,in_channels//2, in_channels,True))
                  down_layer_list.append(ArrayBlock(in_channels,in_channels//2, in_channels,True))
                out_channels=in_channels//2
                down_layer_list.append(down_layer(in_channels,out_channels))
                shape=[out_channels]
            else:
                out_channels=in_channels*2
                for _ in range(residual_blocks):
                  down_layer_list.append(PixelBlock(in_channels,in_channels//2,in_channels,True))
                  down_layer_list.append(PixelBlock(in_channels,in_channels//2,in_channels,True))
                down_layer_list.append(down_layer(in_channels,out_channels,kernel_size,stride,padding=padding))
                shape=(out_channels, *[d//2 for d in shape[1:]])
            down_layer_list.append(nn.LeakyReLU())
            down_layer_list.append(norm(out_channels))
            print("down layer shape ",shape)
        
        for layer in down_layer_list:
            zero_input=layer(zero_input)

        print("down layers zero input",zero_input.size())

        final_down_shape=1
        for n in shape:
            final_down_shape*=n

        print('final_down_shape',final_down_shape)
        up_layer_list=[]
        shape=output_dim
        if output_modality=="voxel":
            up_layer_list.append(nn.Conv3d(4,shape[0],1,1))
        elif output_modality=="pixel":
            up_layer_list.append(nn.Conv2d(4,shape[0],1,1))
            
            shape=(4, *shape[1:])
        print('shape',shape)
        for _ in range(n_layers_trans):
            out_channels=shape[0]
            up_layer_list.append(nn.LeakyReLU())
            if output_modality=="array":
                in_channels=out_channels//2
                
                up_layer_list.append(up_layer(in_channels,out_channels))
                for _ in range(residual_blocks):
                  up_layer_list.append(ArrayBlock(in_channels,in_channels//2, in_channels,True))
                  up_layer_list.append(ArrayBlock(in_channels,in_channels//2, in_channels,True))
                shape=[in_channels]
            else:
                in_channels=out_channels*2
                
                up_layer_list.append(up_layer(in_channels,out_channels,factor,factor))
                for _ in range(residual_blocks):
                  up_layer_list.append(PixelBlock(in_channels,in_channels//2, in_channels,True))
                  up_layer_list.append(PixelBlock(in_channels,in_channels//2, in_channels,True))
                
                shape=(in_channels, *[d//2 for d in shape[1:]])
            print("up layer shape",shape)
        initial_up_shape=1
        for n in shape:
            initial_up_shape*=n

        if output_modality=="pixel" or output_modality=="voxel":
            up_layer_list.append(ReshapeLayer(shape))
        up_layer_list=up_layer_list[::-1]
        if output_modality=="pixel":
            up_layer_list.append(nn.Sigmoid())
        elif output_modality=="array" or output_modality=="voxel":
            up_layer_list.append(nn.Tanh())
        print('initial_up_shape',initial_up_shape)

        intermediate_layers=[]
        if input_modality=="voxel" or input_modality=="pixel":
            intermediate_layers.append(nn.Flatten())

        print("final_down_shape,initial_up_shape",final_down_shape,initial_up_shape)
        intermediate_layers.append(nn.Linear(final_down_shape,initial_up_shape))

        for layer in intermediate_layers:
            zero_input=layer(zero_input)

        print("intermediate zero input",zero_input.size())

        for layer in up_layer_list:
            zero_input=layer(zero_input)

        print("up layers zero input",zero_input.size())

        self.module_list=nn.ModuleList(down_layer_list+intermediate_layers+up_layer_list)


    def forward(self,x):
        for layer in self.module_list:
            x=layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,
                 input_dim,
                 n_layers,
                  input_modality:str, #one of voxel or pixel or array
                  kernel_size:int,
                  factor:int,
                  residual_blocks:int,
                    *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim=input_dim

        

        stride=kernel_size//factor
        padding=stride//factor
        layers=[]
        in_channels=input_dim[0]
        down_layer={
            "voxel":nn.Conv3d,
            "pixel":nn.Conv2d,
            "array":nn.Linear
        }[input_modality]
        norm={
            "voxel":nn.BatchNorm3d,
            "pixel":nn.BatchNorm2d,
            "array":nn.LayerNorm
        }[input_modality]

        for _ in range(n_layers):
            if input_modality=="pixel" or input_modality=="voxel":
                out_channels=in_channels*2
                for _ in range(residual_blocks):
                    layers.append(PixelBlock(in_channels,in_channels//2,in_channels,True))
                layers.append(down_layer(in_channels,out_channels,kernel_size,stride,padding=padding))
            else:

                out_channels=in_channels//2
                for _ in range(residual_blocks):
                    layers.append(ArrayBlock(in_channels,in_channels//2,in_channels,True))
                layers.append(down_layer(in_channels,out_channels))
            layers.append(norm(out_channels))
            layers.append(nn.LeakyReLU())
            in_channels=out_channels
        layers.append(nn.Flatten())
        self.sequential=nn.Sequential(*layers)
        zero_tensor=torch.zeros(input_dim).unsqueeze(0)
        print('zero_tensor.size()',zero_tensor.size())
        zero_output=self.sequential(zero_tensor)
        print('zero output size',zero_output.size())
        dim=1
        for m in zero_output.size():
            dim*=m
        in_channels=2**(1+n_layers)
        self.linear=nn.Linear(dim,1)

        self.layers=nn.ModuleList(layers+[self.linear])

    def forward(self,x):
        x= self.sequential(x)
        #print("self.sequential(x)",x.size())
        x=self.linear(x)
        return x