import torch
from diffusers import DDIMScheduler
from diffusers import StableDiffusionInpaintPipeline

from modules.pipeline_stable_diffusion_inpaint_feature_extraction import StableDiffusionInpaintPipelineFeatureExtraction
from modules.pipeline_max4zero import StableDiffusionInpaintPipelineControlAttention
from modules.max4zero_unet import Max4ZeroUNet2DConditionModel

from config import MEAConfig

def get_feat_extraction_model(pipeline : StableDiffusionInpaintPipeline) -> StableDiffusionInpaintPipeline :
    """
    Load and return the feature extraction model using components from the given pipeline.
    """
    print("Loading Feature Extraction model...")

    pipe = StableDiffusionInpaintPipelineFeatureExtraction(
        vae =               pipeline.vae, 
        text_encoder=       pipeline.text_encoder, 
        tokenizer=          pipeline.tokenizer, 
        unet=               pipeline.unet, 
        scheduler=          pipeline.scheduler, 
        safety_checker=     pipeline.safety_checker,
        feature_extractor=  pipeline.feature_extractor, 
        image_encoder=      pipeline.image_encoder, 
    )

    print("Done.")
    return pipe


def get_inpainting_model(mea_config : MEAConfig, device : str, model : str ="stabilityai/stable-diffusion-2-inpainting",cache_dir : str =None) -> StableDiffusionInpaintPipeline :
    """
    Load and return the inpainting model with the specified MEA configuration.
    """
    print("Loading Stable Diffusion Inpainting model...")
    print(f"Using model name: {model}")

    scheduler = DDIMScheduler.from_config(model, subfolder="scheduler", cache_dir=cache_dir)
    unet = Max4ZeroUNet2DConditionModel.from_pretrained(model, subfolder="unet", cache_dir=cache_dir).to(torch.float16)
   
    pipe = StableDiffusionInpaintPipelineControlAttention.from_pretrained(
        model,
        revision="fp16",
        torch_dtype=torch.float16,
        scheduler=scheduler,
        unet=unet,
        safety_checker=None, 
        mea_config=mea_config, 
        cache_dir=cache_dir
    )

    pipe.to(device)

    print("Done.")
    return pipe

