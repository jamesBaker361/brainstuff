import torch
from torch import nn
from functools import reduce
import operator
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config,GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_outputs import BaseModelOutput,BaseModelOutputWithPastAndCrossAttentions,CausalLMOutputWithCrossAttentions
from transformers.utils import (

    logging,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from typing import Optional,Union,Tuple

logger = logging.get_logger(__name__)
from deep_modeling import PixelBlock,ArrayBlock


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, context, context_mask=None):
        """
        x: [batch, tgt_len, dim]
        context: [batch, ctx_len, dim]
        context_mask: [batch, ctx_len] (0 for pad, 1 for real tokens)
        """
        if context_mask is not None:
            # Convert to key_padding_mask: True for padding, False for valid tokens
            key_padding_mask = context_mask == 0  # shape: [batch, ctx_len]
        else:
            key_padding_mask = None

        attn_out, _ = self.cross_attn(
            query=x,
            key=context,
            value=context,
            key_padding_mask=key_padding_mask
        )

        return self.norm(x + attn_out)





class FMRIConditionedGPT2(nn.Module):
    def __init__(self, fmri_dim=128, model_name='openai-community/gpt2-xl', num_cross_layers=4):
        super().__init__()
        self.gpt2lm=GPT2LMHeadModel.from_pretrained(model_name)
        self.gpt = self.gpt2lm.transformer
        self.config = self.gpt.config
        self.cross_layers = nn.ModuleList()

        # Project fMRI input to GPT embedding space
        self.fmri_proj = nn.Linear(fmri_dim, self.config.hidden_size)

        # Add cross-attention after selected GPT layers
        for i in range(self.config.num_hidden_layers):
            if i==num_cross_layers:
                break
            self.cross_layers.append(CrossAttentionBlock(self.config.hidden_size, self.config.num_attention_heads))


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        fmri_embedding=None,
        labels=None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        fmri_context = self.fmri_proj(fmri_embedding)  # [batch, ctx_len, hidden]
        fmri_context = fmri_context.unsqueeze(1).expand(-1, input_ids.size()[1], -1)
        output_attentions = output_attentions if output_attentions is not None else self.gpt.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.gpt.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.gpt.config.use_cache
        return_dict = return_dict if return_dict is not None else self.gpt.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.gpt.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.gpt.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.gpt.wte(input_ids)
        position_embeds = self.gpt.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        _use_sdpa = self.gpt._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self.gpt._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.gpt.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.gpt.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.gpt.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self.gpt._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.gpt.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.gpt.get_head_mask(head_mask, self.gpt.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.gpt.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.gpt.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gpt.gradient_checkpointing and self.gpt.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.gpt.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(len(self.gpt.h)):
            block, layer_past = self.gpt.h[i], past_key_values[i]
            # Model parallel
            if self.gpt.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gpt.gradient_checkpointing and self.gpt.training:
                outputs = self.gpt._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.gpt.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.gpt.model_parallel:
                for k, v in self.gpt.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.gpt.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.gpt.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        hidden_states = self.gpt.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.gpt2lm.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.gpt2lm.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions
        )
        
        



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