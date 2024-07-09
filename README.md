# MaX4Zero: Masked Extended Attention for Zero-Shot Virtual Try-On In The Wild

> Nadav Orzech, Yotam Nitzan, Ulysse Mizrahi, Dov Danon, Amit H. Bermano
> Tel Aviv University  
>
>Virtual Try-On (VTON) is a highly active line of research, with increasing demand.  It aims to replace a piece of garment in an image with one from another, while preserving person and garment characteristics as well as image fidelity. Current literature takes a supervised approach for the task, impairing generalization and imposing heavy computation. In this paper, we present a novel zero-shot training-free method for inpainting a clothing garment by reference. Our approach employs the prior of a diffusion model with no additional training, fully leveraging its native generalization capabilities. The method employs extended attention to transfer image information from reference to target images, overcoming two significant challenges. We first initially warp the reference garment over the target human using deep features, alleviating "texture sticking". We then leverage the extended attention mechanism with careful masking, eliminating leakage of reference background and unwanted influence. Through a user study, qualitative, and quantitative comparison to state-of-the-art approaches, we demonstrate superior image quality and garment preservation compared unseen clothing pieces or human figures.

<a href="https://arxiv.org/abs/2406.15331"><img src="https://img.shields.io/badge/arXiv-2406.15331-b31b1b.svg" height=22.5></a>
<a href="https://nadavorzech.github.io/max4zero.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=red" height=20.5></a>

<p align="center">
<img src="docs/teaser.png" width="90%"/>  
<br>
MaX4Zero performs Virtual Try-On in the wild (for unseen target images and garments) without any fine-tuning. Given a target image (top) and a garment image (bottom), an image is generated using a diffusion-based prior that replaces the input garment with the one already worn in the target (right).
</p>

## Description  
Official implementation of our MaX4Zero: Masked Extended Attention for Zero-Shot Virtual Try-On In The Wild.

## Environment
Our code builds on the requirement of the `diffusers` library. To set up their environment, please run:
```
conda env create -f environment/environment.yaml
conda activate cross_image
```

## Usage  
To generate an image, you can simply run the `main.py` script. For example,
```
python main.py \
--data.input_dir path/to/max4zero/input/directory \
--data.output_dir path/to/output/directory \
--data.device 0 \
--data.save_stg1_debug_image False \
--mea.mea_scale 15.0 \
--mea.mea_enhance 1.25 \
```

Notes:
- In order to successfully run this repo the input directory structure needs to be a follows:
```
input_directory/
├── images/
│   ├── R_{reference_image_name.png}
│   └── T_{target_image_name.png}
├── masks/
│   ├── R_{reference_mask_name.png}
│   └── T_{target_mask_name.png}
├── neg_masks/
│   └── {neg_mask_name.png} (mask of undesired areas such as free hands, objects intersect with the inpainting mask etc..)
└── prompt.txt (description of the reference garment)
```

## Acknowledgements 
This code builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library. In addition, we 
borrow code from the following repositories: 
- [Cross Image Attention](https://github.com/garibida/cross-image-attention) for computing MEA Enhancement 


## Citation
If you use this code for your research, please cite the following work: 
```
@misc{orzech2024masked,
      title={Masked Extended Attention for Zero-Shot Virtual Try-On In The Wild}, 
      author={Nadav Orzech and Yotam Nitzan and Ulysse Mizrahi and Dov Danon and Amit H. Bermano},
      year={2024},
      eprint={2406.15331},
      archivePrefix={arXiv},
      primaryClass={id='cs.CV' full_name='Computer Vision and Pattern Recognition' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'}
    }
```