from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

@dataclass
class MEAConfig:
    # Masked Extended Attention guidance scale
    mea_scale: float = 15.0  
    
    # Relative timestep where to start using MEA
    t_start: float = 0.7
    # Relative timestep where to end using MEA  
    t_end: float = 1.0  
    # Attention Contrasting strength
    mea_enhance: float = 1.25  
    
    # UNet layers to replace self-attention with MEA
    place_in_unet: List[str] = field(default_factory=lambda: ["up"])  
    # Self-attention layer dimensions replaced with MEA
    dim: List[int] = field(default_factory=lambda: [32, 64])  

    # The following for inner use
    mea_active: bool = field(init=False, default=False)
    inpaint_mask_by_dim: Dict = field(init=False, default_factory=dict)

    def register_inpainting_masks(self, inpaint_masks_dict: Dict):
        """Register inpainting masks for MEA by dimension."""
        self.inpaint_mask_by_dim = inpaint_masks_dict
        
        
@dataclass
class DataConfig:
    # Output base directory
    output_dir: Path = None
    # Stage 1 intermediates results directory name
    stage_1_output_dir: Path = None
    # Stage 2 and final results directory name
    stage_2_output_dir: Path = None
    # Guiding text prompt
    prompt: str = None
    # Input directory including images, masks, and prompt
    input_dir: Path = None
    # Text guiding scale
    scale: float = 12.0  
    # Random seed
    seed: int = 0
    # Device for GPU computing 
    device: int = 0
    # Number of denoising steps
    num_inference_steps: int = 50 
    # Indicates extent to transform the reference image. Must be between 0 and 1.
    strength: float = 1.0
    # Save visualization of matching component
    save_stg1_debug_image: bool = True

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = self.input_dir / "results"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage_1_output_dir = self.output_dir / "stg_1"
        self.stage_2_output_dir = self.output_dir / "stg_2"
        
        self.stage_1_output_dir.mkdir(exist_ok=True)
        self.stage_2_output_dir.mkdir(exist_ok=True)

@dataclass
class Config:
    data : DataConfig = field(default_factory=DataConfig)
    mea : MEAConfig = field(default_factory=MEAConfig)