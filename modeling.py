import torch
from torch import nn

class PixelVoxelModel(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_layers,
                  input_modality:str, #one of voxel or pixel
                  kernel_size:int,
                    *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim=input_dim
        self.output_dim=output_dim
        if type(output_dim)==list or type(output_dim)==tuple:
            self.final_dim=1
            for d in output_dim:
                self.final_dim*=d
        else:
            self.final_dim=output_dim

        stride=kernel_size//2
        layers=[]
        in_channels=input_dim[0]

        conv={
            "voxel":nn.Conv3d,
            "pixel":nn.Conv2d
        }[input_modality]

        for _ in range(n_layers):
            out_channels=in_channels*4
            layers.append(conv(in_channels,out_channels,kernel_size,stride))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.LeakyReLU())
            in_channels=out_channels
        layers.append(nn.Flatten())
        self.sequential=nn.Sequential(*layers)
        zero_tensor=torch.zeros(input_dim).unsqueeze(0)
        zero_output=self.sequential(zero_tensor)
        dim=zero_output.size()[-1]
        self.linear=nn.Linear(dim,self.final_dim)
        self.layers=nn.ModuleList(layers+[nn.Linear])
        

    def forward(self,x):
        batch_size=x.size()[0]
        x= self.sequential(x)
        x=self.linear(x)
        x=x.reshape((batch_size, *self.output_dim))
        return x

class Discriminator(nn.Module):
    def __init__(self,
                 input_dim,
                 n_layers,
                 input_modality:str, #one of voxel or pixel
                  kernel_size:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers=[]
        conv={
            "voxel":nn.Conv3d,
            "pixel":nn.Conv2d
        }[input_modality]
        in_channels=input_dim[0]
        stride=kernel_size//2
        for _ in range(n_layers):
            out_channels=in_channels*4
            layers.append(conv(in_channels,out_channels,kernel_size,stride))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.LeakyReLU())
            in_channels=out_channels