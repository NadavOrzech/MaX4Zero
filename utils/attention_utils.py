from typing import Optional
import torch

from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor

from config import MEAConfig

class MaskedExtAttnProcessor(AttnProcessor):

    def __init__(self, mea_config, place_in_unet, name):
        super().__init__()
        self.mea_config : MEAConfig = mea_config
        self.place_in_unet = place_in_unet
        self.layer_name = name

    r"""
    Masked Extended Attention processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        dim = query.shape[1] ** 0.5
        is_cross = encoder_hidden_states.shape[1] == 77

        # Check if use MEA in this case
        mea_cond = not is_cross and self.mea_config.mea_active and self.place_in_unet in self.mea_config.place_in_unet and dim in self.mea_config.dim

        if mea_cond:
            hidden_states = self.masked_extended_attention(query, key, value, batch_size, attn, attention_mask)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
       
       
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


    def attention_function(self, 
                            query : torch.Tensor, 
                            key : torch.Tensor, 
                            value : torch.Tensor,
                            scale : float,
                            attention_mask=None,
                            ):
        
        dtype = query.dtype
        device = query.device

        beta = 0
        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], 
                query.shape[1], 
                key.shape[1], 
                dtype=dtype, 
                device=device
            )
        else:
            baddbmm_input = (1.-attention_mask) * -10_000.0
            beta = 1

        attention_scores = torch.baddbmm(   
            baddbmm_input.to(device),
            query.to(device),
            key.transpose(-1,-2).to(device),
            beta=beta,
            alpha=scale,
        )
        del baddbmm_input

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        if attention_probs.shape[2] != attention_probs.shape[1] and self.mea_config.mea_enhance > 1:
            # Use contrasting operator if in MEA and mea_enhance > 1
            attn_prob_self, attn_prob_next = attention_probs.chunk(2, -1)
            attn_prob_next = torch.stack([
                        torch.clip(
                            self.enhance_tensor(attn_prob_next[head_idx], contrast_factor=self.mea_config.mea_enhance)
                            , min=0.0, max=1.0)
                        for head_idx in range(attn_prob_next.shape[0])
                        ])
            attention_probs = torch.concat((attn_prob_self, attn_prob_next), -1)    


        attention_probs = attention_probs.to(dtype)
        hidden_states = torch.bmm(attention_probs, value.to(device))

        return hidden_states



    def masked_extended_attention(self, 
                              queries : torch.Tensor, 
                              keys : torch.Tensor, 
                              values : torch.Tensor,
                              batch_size: int, 
                              attn: Attention,
                              attention_mask=None,
                              ):
        
        hidden_state_list = []     
        
        ref_q, tar_q = queries.chunk(batch_size)
        ref_k, tar_k = keys.chunk(batch_size)
        ref_v, tar_v = values.chunk(batch_size)
        ref_mask, tar_mask = self.mea_config.inpaint_mask_by_dim[ref_q.shape[1]].chunk(batch_size)                            

        # Get hidden states for the reference image using naive self-attention
        attention_probs_bg = attn.get_attention_scores(query=ref_q, key=ref_k, attention_mask=None)
        hidden_states_bg = torch.bmm(attention_probs_bg, ref_v)
        hidden_state_list.append(hidden_states_bg)

        # Get hidden states for the target image using Masked Extended Attention over reference image
        # Create Extended Attention Mask
        tar_keys = torch.cat([tar_k, ref_k], dim=1)
        tar_values = torch.cat([tar_v, ref_v], dim=1)
        kv_mask = torch.cat([torch.ones_like(tar_mask), ref_mask], dim=1)
        attention_mask = self.create_attention_mask(tar_mask, kv_mask)

        # Get hidden states for target image only inside inpainting mask using MEA
        hidden_states_fg = self.attention_function(query=tar_q, key=tar_keys, value=tar_values, scale=attn.scale, attention_mask=attention_mask)

        # Get hidden states for target image only outside inpainting mask using naive self-attention
        attention_probs_bg = attn.get_attention_scores(query=tar_q, key=tar_k, attention_mask=None)
        hidden_states_bg = torch.bmm(attention_probs_bg, tar_v)

        # Get final hidden state of both inside and outside the inpainting mask
        hidden_states = hidden_states_fg * tar_mask + hidden_states_bg * (1-tar_mask)

        hidden_state_list.append(hidden_states)
        hidden_state = torch.cat(hidden_state_list, dim=0)

        return hidden_state
    
    
    def create_attention_mask(self, q_mask, kv_mask):
        """Creating attention mask for MEA"""
        q_mask = q_mask.squeeze()
        kv_mask = kv_mask.squeeze()
        
        attention_mask = (q_mask.unsqueeze(1) + kv_mask.unsqueeze(0)) > 1
        attention_mask = attention_mask.unsqueeze(0).to(q_mask.device)
        attention_mask = attention_mask.to(q_mask.dtype)
        
        return attention_mask
    
    @staticmethod
    def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
        """ 
        Compute the attention map contrasting. 
        Taken from https://github.com/garibida/cross-image-attention
        """
        adjusted_tensor = (tensor - tensor.mean(dim=-1).unsqueeze(-1)) * contrast_factor + tensor.mean(dim=-1).unsqueeze(-1)
        return adjusted_tensor
    
