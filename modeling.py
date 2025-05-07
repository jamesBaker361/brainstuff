import torch
from torch import nn
from functools import reduce
import operator

def compute_input_size(output_size, n_layers, kernel_size=4, stride=2,):
    factor=kernel_size//stride
    dim=output_size[-1]
    for _ in range(n_layers):
        dim=dim//factor
    return dim,dim


def compute_input_size_3d(output_size, n_layers,
                          kernel_size=4, stride=2):
    factor=kernel_size//stride
    dim=output_size[-1]
    for _ in range(n_layers):
        dim=dim//factor
    return dim,dim,dim

class PixelVoxelModel(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_layers,
                 n_layers_trans:int,
                  input_modality:str, #one of voxel or pixel
                  kernel_size:int,
                    *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim=input_dim
        self.output_dim=output_dim

        

        stride=kernel_size//2
        layers=[]
        in_channels=input_dim[0]
        size_function={
            "voxel":compute_input_size,
            "pixel":compute_input_size_3d
        }[input_modality]
        down_layer={
            "voxel":nn.Conv3d,
            "pixel":nn.Conv2d
        }[input_modality]
        batch={
            "voxel":nn.BatchNorm3d,
            "pixel":nn.BatchNorm2d
        }[input_modality]
        up_layer={
            "voxel":nn.ConvTranspose2d,
            "pixel":nn.ConvTranspose3d
        }[input_modality]

        self.intermediate_dim=size_function(output_dim[1:],n_layers_trans,kernel_size,stride)

        for _ in range(n_layers):
            out_channels=in_channels*2
            layers.append(down_layer(in_channels,out_channels,kernel_size,stride))
            layers.append(batch(out_channels))
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
        in_channels=pow(2,1+n_layers_trans)
        print(self.intermediate_dim)
        flat_intermediate_dim=1
        for n in self.intermediate_dim:
            flat_intermediate_dim*=n
        #print(dim,in_channels,reduce(operator.mul, self.intermediate_dim, 1),in_channels*reduce(operator.mul, self.intermediate_dim, 1))
        print(dim,in_channels,flat_intermediate_dim,in_channels*flat_intermediate_dim)
        self.linear=nn.Linear(dim,in_channels* flat_intermediate_dim)

        trans_layers=[]
        for _ in range(n_layers_trans):
            out_channels=out_channels*2
            trans_layers.append(up_layer(in_channels,out_channels,kernel_size,stride))
            trans_layers.append(nn.LeakyReLU())
            in_channels=out_channels

        #self.final_conv=conv_trans(in_channels,output_dim[1],1,1)

        trans_layers.append(up_layer(in_channels,output_dim[0],1,1))
        self.trans_seqential=nn.Sequential(*trans_layers)

        self.layers=nn.ModuleList(layers+[self.linear]+trans_layers)
        

    def forward(self,x):
        batch_size=x.size()[0]
        x= self.sequential(x)
        print("self.sequential(x)",x.size())
        x=self.linear(x)
        print("self.linear(x)",x.size())
        x=x.reshape((batch_size,-1, *self.intermediate_dim))
        print("x.reshape((batch_size,-1, *self.intermediate_dim))",x.size())
        x=self.trans_seqential(x)
        print("self.trans_seqential(x)",x.size())
        return x

class Discriminator(nn.Module):
    def __init__(self,
                 input_dim,
                 n_layers,
                  input_modality:str, #one of voxel or pixel
                  kernel_size:int,
                    *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim=input_dim

        

        stride=kernel_size//2
        layers=[]
        in_channels=input_dim[0]
        conv={
            "voxel":nn.Conv3d,
            "pixel":nn.Conv2d
        }[input_modality]
        batch={
            "voxel":nn.BatchNorm3d,
            "pixel":nn.BatchNorm2d
        }[input_modality]

        for _ in range(n_layers):
            out_channels=in_channels*2
            layers.append(conv(in_channels,out_channels,kernel_size,stride))
            layers.append(batch(out_channels))
            layers.append(nn.LeakyReLU())
            in_channels=out_channels
        layers.append(nn.Flatten())
        self.sequential=nn.Sequential(*layers)
        zero_tensor=torch.zeros(input_dim).unsqueeze(0)
        print('zero_tensor.size()',zero_tensor.size())
        zero_output=self.sequential(zero_tensor)
        print('zero output size',zero_output.size())
        dim=reduce(operator.mul,zero_output.size(),1)
        in_channels=2**(1+n_layers)
        self.linear=nn.Linear(dim,1)

        self.layers=nn.ModuleList(layers+[self.linear])

    def forward(self,x):
        x= self.sequential(x)
        #print("self.sequential(x)",x.size())
        x=self.linear(x)
        return x