from typing import Callable, List, Optional, Union, Dict, Any, Tuple

import PIL
import torch
import numpy as np 
import cv2
from PIL import Image

from config import Config
from utils.model_utils import get_inpainting_model
from utils.attention_utils import MaskedExtAttnProcessor
from utils.file_utils import *



class ConsistentInpainter():
    def __init__(self, config : Config) -> None:
        
        self.config = config
        self.mea_config = config.mea
        self.pipe = get_inpainting_model(mea_config=config.mea, device=config.data.device)
        
        self._register_attention_control(config.mea)


    def run_inpainting(self,
                       images_batch: List[PIL.Image.Image],
                       masks_batch: List[PIL.Image.Image],
                       prompt_batch: List[str],
                       fringe_mask=None,
                       dilated_fringe_mask=None):

        self._register_mask_patch(masks_batch)
        masks_batch[0] = PIL.Image.new("L",(512,512), color="black")

        images = self.pipe(prompt=prompt_batch,
                            image=images_batch,
                            mask_image=masks_batch,   
                            latents=None,
                            strength= self.config.data.strength,
                            guidance_scale=self.config.data.scale,
                            mea_guidance_scale= self.config.mea.mea_scale,
                            num_inference_steps=self.config.data.num_inference_steps,
                            mea_start= self.config.mea.t_start,
                            mea_end= self.config.mea.t_end,
                            keep_reference_image=True,
                            fringe_masks=[fringe_mask, dilated_fringe_mask],
                            ).images


        target = self._repaint(images[-1], images_batch[-1], masks_batch[-1])
        return target  

    def _register_attention_control(self, mea_cfg):        
        attn_procs = {}   
        for name in self.pipe.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            attn_procs[name] = MaskedExtAttnProcessor(
                mea_config=mea_cfg,
                place_in_unet=place_in_unet, 
                name=name, 
            )
            
        self.pipe.unet.set_attn_processor(attn_procs)
        

    def _register_mask_patch(self, mask_list: List[PIL.Image.Image]):
        mask = torch.tensor(np.concatenate([pil_to_numpy(m.convert("L"))[None, None, :] for m in mask_list], axis=0)).float() / 255.0
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Registering masks by layer shape
        batch, channel, width, height = mask.shape
        resized_masks_dict = {}
        for i in [1, 2, 4, 8]:
            scaled_width = width // (i * self.pipe.vae_scale_factor)
            scaled_height = height // (i * self.pipe.vae_scale_factor)
            resized_tensor = torch.nn.functional.interpolate(
                mask, size=(scaled_width, scaled_height)
            )
            reshaped_tensor = resized_tensor.view(batch, scaled_width * scaled_height, channel)[:batch]
            reshaped_tensor = reshaped_tensor.to(torch.float16).to(self.config.data.device)
            resized_masks_dict[scaled_width * scaled_height] = reshaped_tensor

        self.mea_config.register_inpainting_masks(resized_masks_dict)


    def _repaint(self,
                inpainted_image : Union[Image.Image, np.ndarray], 
                orig_image : Union[Image.Image, np.ndarray], 
                mask : Union[Image.Image, np.ndarray], 
                ) -> Image.Image:

        inpainted_image = to_array(inpainted_image)
        orig_image = to_array(orig_image)
        mask = to_array(mask)

        blurred_mask = cv2.GaussianBlur(mask, (3, 3), 0)
        edges_mask = (blurred_mask < 255) & (blurred_mask > 0)
        blurred_inpainted_image = cv2.GaussianBlur(inpainted_image, (3, 3), 0)
        inpainted_image[mask==0] = orig_image[mask==0]
        inpainted_image[edges_mask] = blurred_inpainted_image[edges_mask]
    
        return Image.fromarray(inpainted_image)