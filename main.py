import cv2
import pyrallis
from PIL import Image

from diffusers.training_utils import set_seed

from config import Config
from modules.matcher import FeatMatcher, FeatMatchDebuger
from utils.file_utils import *
from modules.inpainter import ConsistentInpainter

import warnings
warnings.filterwarnings("ignore")


def create_fringes_mask(original_mask : Image.Image, 
                          deformed_mask : Image.Image, 
                          original_neg_mask : Image.Image = None):
    
    def preprocess_mask(mask: Image.Image) -> np.ndarray:
        mask_array = pil_to_numpy(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        return mask_array
    
    original_mask_array = preprocess_mask(original_mask)
    deformed_mask_array = preprocess_mask(deformed_mask)

    fringes_mask = np.zeros(original_mask.size)
    fringes_mask[original_mask_array >= 125] += 1
    fringes_mask[deformed_mask_array >= 125] -= 1
    fringes_mask[fringes_mask<0] = 0
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19,19))
    dilated_mask = cv2.dilate(fringes_mask, kernel)
    dilated_mask = (dilated_mask >= 0.5) * 255

    # if original_neg_mask is not None:
    #     original_neg_mask_array = preprocess_mask(original_neg_mask)
    #     dilated_mask[original_neg_mask_array == 255] = 0

    dilated_mask[original_mask_array==0]=0
    
    fringes_mask_image = Image.fromarray((fringes_mask * 255).astype(np.int32)).convert('RGB')
    dilated_mask_image = Image.fromarray(dilated_mask.astype(np.int32)).convert('RGB')

    return fringes_mask_image, dilated_mask_image

def run_max4zero(cfg : Config, 
                 inpainter : ConsistentInpainter, 
                 matcher : FeatMatcher):
    
    set_seed(cfg.data.seed)
    pyrallis.dump(cfg, open(cfg.data.stage_1_output_dir / 'config.yaml', 'w'))

    assert cfg.data.input_dir is not None

    input_dir_data = load_inpaint_dir(cfg.data.input_dir)
    save_source_files(input_dir_data, cfg.data.stage_1_output_dir)
    original_target_image = input_dir_data.target_image
    
    matcher.fit(*input_dir_data.get_inpainting_data())
    matching_result = matcher.compute_matching(input_dir_data.reference_mask, 
                                                input_dir_data.target_mask, 
                                                input_dir_data.reference_image, 
                                                input_dir_data.target_image, 
                                                )

    fringes_mask_image, dilated_fringes_mask_image = create_fringes_mask(input_dir_data.target_mask, matching_result["deformed_mask"], input_dir_data.target_neg_mask)
    input_dir_data.update_target(fringes_mask=fringes_mask_image, target_image=matching_result["x_g_p"])


    if cfg.data.save_stg1_debug_image:
        dbg_output_dir = cfg.data.stage_1_output_dir / "DBG"
        dbg_output_dir.mkdir(exist_ok=True)
        DebugImage({
            "ref image": input_dir_data.reference_image,
            "target image": input_dir_data.target_image,
            "matching": matching_result["visualization"],
            "ref on target": matching_result["x_g_p"],
        }).save_grid(dbg_output_dir / "matching.png")

    
    save_source_files(input_dir_data, cfg.data.stage_2_output_dir)
    
    target = inpainter.run_inpainting(*input_dir_data.get_inpainting_data(), 
                                      fringe_mask=fringes_mask_image, 
                                      dilated_fringe_mask=dilated_fringes_mask_image
                                      )
    target.save(cfg.data.stage_2_output_dir / (input_dir_data.target_name + f"_out.png"))


@pyrallis.wrap()
def main(cfg : Config):

    inpainter = ConsistentInpainter(cfg)

    if cfg.data.save_stg1_debug_image:
        matcher = FeatMatchDebuger(device=cfg.data.device, prev_pipeline=inpainter.pipe)
    else:
        matcher = FeatMatcher(device=cfg.data.device, prev_pipeline=inpainter.pipe)

    run_max4zero(cfg, inpainter, matcher)
    
if __name__ == "__main__":


    main()
