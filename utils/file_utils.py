from typing import List, Union, Tuple, Dict
from PIL import Image
import numpy as np 
import json
from pathlib import Path
from dataclasses import dataclass, field
import cv2

@dataclass
class InputData:
    """Dataclass holding all input objects for the Max4zero 2 stages pipeline"""
    reference_name: str
    reference_image: Image.Image
    reference_mask: Image.Image
    ref_description: str
    
    target_name: str
    target_image: Image.Image
    target_mask: Image.Image
    target_neg_mask : Image.Image

    fringes_mask : Image.Image = None

    def get_inpainting_data(self) -> Tuple[List[Image.Image], List[Image.Image], List[str]]:
        """Get data for inpainting"""
        return  [self.reference_image, self.target_image], \
                [self.reference_mask, self.target_mask], \
                [self.ref_description, self.ref_description]
       
    
    def save_to_directory(self, directory_path: str) -> None:
        """Save source reference and target data to a directory."""
        directory = Path(directory_path)
        directory.mkdir(parents=True, exist_ok=True)
        
        self._save_image(self.reference_image, directory, f"{self.reference_name}.png")
        self._save_image(self.reference_mask, directory, f"{self.reference_name}_mask.png")
        self._save_image(self.target_image, directory, f"{self.target_name}.png")
        self._save_image(self.target_mask, directory, f"{self.target_name}_mask.png")
        
        if self.fringes_mask:
            self._save_image(self.fringes_mask, directory, f"{self.target_name}_edges_mask.png")

        desc_file_path = directory / "reference_desc.txt"

        save_text(desc_file_path, self.ref_description)

    def update_target(self, target_name=None, target_image=None, target_mask=None, fringes_mask=None):
        """Update target attributes."""
        if target_name is not None:
            self.target_name = target_name
    
        if target_image is not None:
            self.target_image = target_image

        if target_mask is not None:
            self.target_mask = target_mask
        
        if fringes_mask:
            self.fringes_mask = fringes_mask

    def _save_image(self, image, directory, filename):
        """Save an image to the directory."""
        image_path = directory / filename
        image.save(image_path)

@dataclass
class DebugImage:
    """Dataclass holding debug objects stage 1 visualization images"""
    grid_image_list: Dict[str, Image.Image] = field(default_factory=dict) 

    def save_grid(self, output_path: str) -> None:
        """Save the grid of debug images."""
        res = [self.text_under_image(pil_to_numpy(value), key) for key, value in self.grid_image_list.items()]
        res = np.concatenate(res, 1)
        Image.fromarray(res).save(output_path)

    def text_under_image(self, image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0), font_size: int = 1, thickness : int = 2):
        h, w, c = image.shape
        offset = int(h * .2)
        img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
        img[:h] = image
        textsize = cv2.getTextSize(text, font, font_size, thickness)[0]
        text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
        cv2.putText(img, text, (text_x, text_y ), font, font_size, text_color, thickness)
        return img
    

def load_text(file_path: Path) -> str:
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except IOError as e:
        print(f'Error reading from {file_path}: {e}')
        return None
    
def save_text(file_path: Path, content: str) -> None:
    try:
        with open(file_path, 'w') as file:
            file.write(content)
    except IOError as e:
        print(f'Error saving to {file_path}: {e}')

def load_json(json_path: Path) -> dict:
    with open(json_path, "r") as reader:
        return json.load(reader)

def save_json(data: dict, json_path: Path) -> None:
    with open(json_path, "w") as writer:
        json.dump(data, writer)

def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    return np.array(pil_image, copy=True)

def to_array(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    return pil_to_numpy(image) if isinstance(image, Image.Image) else image

def get_all_images_from_dir(dir_name: str, files_suffix: Union[str, List[str]] = ["jpg", "png"]) -> List[str]:
    dir_path = Path(dir_name)
    if isinstance(files_suffix, str):
        files_suffix = [files_suffix]

    image_paths = []
    for suf in files_suffix:
        if '.' in suf:
            image_paths.extend(dir_path.glob(f"*{suf}"))
        else:
            image_paths.extend(dir_path.glob(f"*.{suf.lower()}"))
    
    image_paths = sorted(image_paths)

    return [str(path) for path in image_paths]

def load_image(image_path: str, resize_image_dims: Tuple[int, int] = (512, 512), crop_square: bool = False, gray: bool = False) -> Image.Image:
    """Load and process an image.

    Args:
        image_path (str): Path to the image file.
        resize_image_dims (Tuple[int, int], optional): Dimensions to resize the image. Defaults to (512, 512).
        crop_square (bool, optional): Whether to crop the image to a square. Defaults to False.
        gray (bool, optional): Whether to convert the image to grayscale. Defaults to False.

    Returns:
        Image.Image: Processed image.
    """
    
    if gray:
        image = pil_to_numpy(Image.open(image_path).convert("L"))
        image = (image >= 125).astype(np.uint8) * 255
        image = np.stack([image] * 3, axis=-1)
    else:
        image = pil_to_numpy(Image.open(image_path).convert("RGB"))
    
    if crop_square:
        h, w = image.shape[:2]       
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        bottom = top + min_dim
        left = (w - min_dim) // 2
        right = left + min_dim
        image = image[top:bottom, left:right]

    image = Image.fromarray(image)
    if resize_image_dims:
        image = image.resize(resize_image_dims, resample=Image.Resampling.BICUBIC)
        
    return image

def load_inpaint_dir(input_dir: Path, target_mask_dil: int = 13) -> InputData:
    # Get image paths
    # Images directory needs to be with reference image named 'R_*' and target image named 'T_*'
    # Masks directory needs to be with reference mask named 'R_*' and mask image named 'T_*'
    ref_img_path, target_img_path = get_all_images_from_dir(dir_name=(input_dir / "images"))
    ref_mask_path, target_mask_path = get_all_images_from_dir(dir_name=(input_dir / "masks"))
    
    # Load prompt
    prompt_file = input_dir / "prompt.txt"
    prompt = load_text(prompt_file)

    # Load and process reference mask
    ref_mask = load_image(ref_mask_path, crop_square=True, gray=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated_mask = 255 - cv2.dilate(255 - pil_to_numpy(ref_mask), kernel)
    ref_mask = Image.fromarray(dilated_mask)

    # Load and process target mask
    target_mask = load_image(target_mask_path, crop_square=True, gray=True)
    target_mask = pil_to_numpy(target_mask)

    # Load and process unwanted areas mask if available (i.e hands / other blocking objects)
    neg_mask = None
    neg_mask_path = None
    neg_masks_dir = input_dir / "neg_masks"
    if neg_masks_dir.is_dir():
        neg_mask_paths = get_all_images_from_dir(neg_masks_dir)
        if neg_mask_paths:
            neg_mask_path = neg_mask_paths[0]
            neg_mask = load_image(neg_mask_path, crop_square=True, gray=True)
            target_mask[pil_to_numpy(neg_mask) == 255] = 0

    
    # Dilate target mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (target_mask_dil,target_mask_dil))
    dilated_target_mask = cv2.dilate(target_mask, kernel)
    target_mask = Image.fromarray(dilated_target_mask)

    # Load images
    reference_image=load_image(ref_img_path, crop_square=True)
    target_image=load_image(target_img_path, crop_square=True)

    # Return populated InputData object
    return InputData(
        reference_name=Path(ref_img_path).stem, 
        reference_image=reference_image,
        reference_mask=ref_mask,
        ref_description=prompt,

        target_name=Path(target_img_path).stem, 
        target_image=target_image,
        target_mask=target_mask,
        target_neg_mask=neg_mask
    )

def save_source_files(inpaint_input: InputData, output_dir: Path) -> None:
    soruce_output_dir = output_dir / "source"
    soruce_output_dir.mkdir(exist_ok=True)
    inpaint_input.save_to_directory(soruce_output_dir)
