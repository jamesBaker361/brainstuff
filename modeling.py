import torch
from torch import nn
from functools import reduce
import operator
from deep_modeling import ArrayBlock, PixelBlock
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import LlamaForCausalLM, LlamaTokenizer

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, context):
        attn_out, _ = self.cross_attn(x, context, context)
        return self.norm(x + attn_out)

class FMRIConditionedGPT2(nn.Module):
    def __init__(self, fmri_dim=128, model_name='gpt2', num_cross_layers=4):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.config = self.gpt2.config
        self.cross_layers = nn.ModuleList()

        # Project fMRI input to GPT embedding space
        self.fmri_proj = nn.Linear(fmri_dim, self.config.hidden_size)

        # Add cross-attention after selected GPT layers
        for i in range(self.config.n_layer):
            if i < num_cross_layers:
                self.cross_layers.append(CrossAttentionBlock(self.config.hidden_size, self.config.n_head))
            else:
                self.cross_layers.append(None)

    def forward(self, input_ids, fmri_embed):
        # Project and expand fMRI to [batch, seq_len=1, hidden_size]
        B = self.fmri_proj(fmri_embed)  # [batch, seq_len, D]
        gpt_outputs = self.gpt2.wte(input_ids)  # [batch, seq, D]
        hidden_states = gpt_outputs

        for i, block in enumerate(self.gpt2.h):
            hidden_states = block(hidden_states)[0]  # GPT-2 self-attn + FFN
            if self.cross_layers[i] is not None:
                hidden_states = self.cross_layers[i](hidden_states, B)

        return hidden_states


from transformers import LlamaForCausalLM, LlamaTokenizer

class FMRIConditionedLLaMA(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", fmri_dim=128, num_cross_layers=-1):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(model_name)
        self.config = self.model.config

        self.fmri_proj = nn.Linear(fmri_dim, self.config.hidden_size)
        self.cross_layers = nn.ModuleList()

        for i in range(self.config.num_hidden_layers):
            if i==num_cross_layers:
                break
            self.cross_layers.append(CrossAttentionBlock(self.config.hidden_size, self.config.num_attention_heads))
            
        print(f"added {len(self.cross_layers)} cross_layers")
        self._insert_cross_attn_hooks()

    def _insert_cross_attn_hooks(self):
        for i, block in enumerate(self.model.model.layers):
            if self.cross_layers[i] is not None:
                old_forward = block.forward

                def make_forward(old_forward, cross_attn):
                    def new_forward(*args, **kwargs):
                        hidden_states = args[0]
                        outputs = old_forward(*args, **kwargs)
                        hidden_states = outputs[0]
                        fmri_context = kwargs.get("fmri_context")
                        if fmri_context is not None:
                            hidden_states = cross_attn(hidden_states, fmri_context)
                        return (hidden_states,) + outputs[1:]
                    return new_forward

                block.forward = make_forward(old_forward, self.cross_layers[i])

    def forward(self, input_ids, attention_mask, fmri_embedding):
        fmri_context = self.fmri_proj(fmri_embedding)  # [batch, ctx_len, hidden]
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          fmri_context=fmri_context)


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
                  #down_layer_list.append(ArrayBlock(in_channels,in_channels//2, in_channels,True))
                  down_layer_list.append(ArrayBlock(in_channels,in_channels//2, in_channels,True))
                out_channels=in_channels//2
                down_layer_list.append(down_layer(in_channels,out_channels))
                shape=[out_channels]
            else:
                out_channels=in_channels*2
                for _ in range(residual_blocks):
                  #down_layer_list.append(PixelBlock(in_channels,in_channels//2,in_channels,True))
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
                  #up_layer_list.append(ArrayBlock(in_channels,in_channels//2, in_channels,True))
                  up_layer_list.append(ArrayBlock(in_channels,in_channels//2, in_channels,True))
                shape=[in_channels]
            else:
                in_channels=out_channels*2
                
                up_layer_list.append(up_layer(in_channels,out_channels,factor,factor))
                for _ in range(residual_blocks):
                  #up_layer_list.append(PixelBlock(in_channels,in_channels//2, in_channels,True))
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
    
class FusedModel(nn.Module):
    def __init__(self, pixel_to_fmri,fmri_to_pixel,pixel_discriminator=None,fmri_discriminator=None):
        super().__init__()
        self.pixel_to_fmri=pixel_to_fmri
        self.fmri_to_pixel=fmri_to_pixel
        self.pixel_discriminator=pixel_discriminator
        self.fmri_discriminator=fmri_discriminator


class SuperResolutionModel(nn.Module):
    def __init__(self,input_dim,
                 output_dim,
                 residual_blocks,*args, **kwargs):
        super().__init__(*args, **kwargs)
        layer_list=[]
        in_channels=input_dim[0]
        out_channels=output_dim[0]
        in_width=input_dim[-1]
        out_width=output_dim[-1]

        layers=0
        while out_width>in_width:
            layers+=1
            out_width=out_width//2

        for _ in range(layers):
            for __ in range(residual_blocks):
                layer_list.append(PixelBlock(in_channels,in_channels//2,in_channels,True))
            layer_list.append(nn.ConvTranspose2d(in_channels,in_channels//2,2,2))
            in_channels=in_channels//2
        layer_list.append(nn.Conv2d(in_channels,out_channels,1,1))
        layer_list.append(nn.Sigmoid())

        self.module_list=nn.ModuleList(layer_list)

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