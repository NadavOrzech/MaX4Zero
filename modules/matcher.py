
import cv2
import torch
import numpy as np 
import PIL
from PIL import Image
from typing import List

from utils.model_utils import get_feat_extraction_model
from utils.file_utils import *

class FeatMatcher():
    def __init__(self, device, prev_pipeline=None) -> None:
        self.pipe = get_feat_extraction_model(prev_pipeline)
        self.features = None
        self.device = device

    def fit(self, 
            images_batch : List[PIL.Image.Image], 
            masks_batch : List[PIL.Image.Image],  
            prompt_batch : List[str], 
            feat_extraction_layer_index = 1
            ):
        
        # TODO: increase batch by multiply images for better accuracy
        
        ft_out = self.pipe(
                    prompt=prompt_batch, 
                    image=images_batch, 
                    mask_image=masks_batch,                      
                    strength=1,               
                    guidance_scale=1, 
                    num_inference_steps=50,
                    extract_feat_indices = [feat_extraction_layer_index],
                    denoise_index_for_feat_extraction=0
                    )  
        
        self.features = ft_out['up_ft'][feat_extraction_layer_index]
       
    def _clear_features(self):
        del self.features
        self.features = None
        torch.cuda.empty_cache()


    def compute_matching(self,
                         ref_mask : PIL.Image.Image, 
                         target_mask : PIL.Image.Image, 
                         ref_img_base : PIL.Image.Image = None, 
                         target_img_base : PIL.Image.Image = None,
                         return_debug_images : bool = False, 
                         ):
        ref_img_base = to_array(ref_img_base.convert("RGB"))
        target_img_base = to_array(target_img_base.convert("RGB"))
        ref_mask = to_array(ref_mask.convert("L"))
        target_mask = to_array(target_mask.convert("L"))

        ref_mask[ref_mask<125] = 0
        ref_mask[ref_mask>=125] = 255
        target_mask[target_mask<125] = 0
        target_mask[target_mask>=125] = 255

        assert return_debug_images is None or \
            (return_debug_images is not None and ref_img_base is not None and target_img_base is not None)
        
        
        batch, channel, width, height = self.features.shape

        ref_feat, target_feat = self.features.chunk(batch)

        reshaped_ref_mask = FeatMatcher._resize_mask(ref_mask, width)
        reshaped_target_mask = FeatMatcher._resize_mask(target_mask, width)
        
        matching_pixels, matching_matrix = FeatMatcher._mnn_matcher(ref_feat, target_feat, reshaped_ref_mask, reshaped_target_mask)
        
        src_pixels, dst_pixels = FeatMatcher._filter_matches_arrays(matching_pixels)
        deformed_ref_image, deformed_ref_mask = FeatMatcher._deform_image(ref_img_base, ref_mask, src_pixels, dst_pixels)
        ref_on_target = np.where(deformed_ref_mask * np.tile(target_mask[...,None], 3) == 1, deformed_ref_image, target_img_base)

        ref_on_target = Image.fromarray(ref_on_target)
        self._clear_features()
        return {
                "x_g_p": ref_on_target,
                "deformed_mask": deformed_ref_mask
                }
    
    @staticmethod
    def _remove_outliers(points1, points2):
        _, inliers = cv2.findHomography(points1[:, :-1], points2[:, :-1], cv2.RANSAC, 5)
        return  [i for i in range(len(points1)) if inliers[i] == 1]

    
    @staticmethod
    def _filter_matches_arrays(pixel_array):
        src_pixels = pixel_array[0, :, 1:] 
        dst_pixels = pixel_array[1, :, 1:]
        good_indices = FeatMatcher._remove_outliers(src_pixels, dst_pixels)

        filterd_src_pixels = src_pixels[good_indices]
        filterd_dst_pixels = dst_pixels[good_indices]

        return filterd_src_pixels, filterd_dst_pixels
    
    @staticmethod
    def _mnn_matcher(descriptors_a, descriptors_b, mask1, mask2):
        
        num_images, num_channel, dim, _ = descriptors_b.shape
        
        descriptors_a = descriptors_a.view(1, num_channel, -1)
        descriptors_b = descriptors_b.view(num_images, num_channel, -1)
        
        
        # cosine similarity
        descriptors_a = torch.nn.functional.normalize(descriptors_a)
        descriptors_b = torch.nn.functional.normalize(descriptors_b)
        sim = torch.matmul(descriptors_a.transpose(1,2), descriptors_b)
        sim = sim.transpose(0,1).reshape(sim.shape[1], -1)
        
        sim_mask = torch.zeros_like(sim)
        sim_mask[mask1 == 1] += 1
        sim_mask.transpose(0,1)[mask2 == 1] += 1
        sim_mask[sim_mask < 2] = 0
        sim_mask[sim_mask == 2] = 1

        sim_min = sim.min()
        sim[sim_mask==0] = sim_min

       
        matching_matrix, matching_pixels = FeatMatcher._bilinear_best_match(
                                                                        sim, 
                                                                        ax1_num_images=1,
                                                                        ax2_num_images=num_images, 
                                                                        return_as_pixels=True
                                                                        ) 
        
        return matching_pixels, matching_matrix   

    @staticmethod
    def _bilinear_best_match(matrix, ax1_num_images=1, ax2_num_images=1, return_as_pixels=False):
        nn12_v, nn12 = torch.max(matrix, dim=1)
        nn21_v, nn21 = torch.max(matrix, dim=0)
        
        image_dim = int(matrix.shape[0] ** 0.5)
        
        ids1 = torch.arange(0, matrix.shape[0], device=matrix.device)
        mask = (ids1 == nn21[nn12])

    
        matching_matrix = torch.zeros_like(matrix, device="cpu")
        matching_matrix[ids1[mask], nn12[mask]] = 1

        matching_pixels = None
        if return_as_pixels:
            pixels1 = np.column_stack(np.unravel_index(ids1[mask].cpu(), (ax1_num_images, image_dim, image_dim)))
            pixels2 = np.column_stack(np.unravel_index(nn12[mask].cpu(), (ax2_num_images, image_dim, image_dim)))

            pixels1 = np.concatenate((pixels1, nn12_v[ids1[mask]].cpu().unsqueeze(-1)), -1)
            pixels2 = np.concatenate((pixels2, nn21_v[nn12[mask]].cpu().unsqueeze(-1)), -1)

            matching_pixels = np.stack([pixels1, pixels2])
            
             
        return matching_matrix, matching_pixels
    
    @staticmethod
    def _mls_affine_deformation(vy, vx, p, q, alpha=1.0, eps=1e-8):
        """
        Affine deformation

        Parameters
        ----------
        vy, vx: ndarray
            coordinate grid, generated by np.meshgrid(gridX, gridY)
        p: ndarray
            an array with size [n, 2], original control points, in (y, x) formats
        q: ndarray
            an array with size [n, 2], final control points, in (y, x) formats
        alpha: float
            parameter used by weights
        eps: float
            epsilon
        
        Return
        ------
            A deformed image.
        """

        # Change (x, y) to (row, col)
        q = np.ascontiguousarray(q.astype(np.int16))
        p = np.ascontiguousarray(p.astype(np.int16))

        # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
        p, q = q, p

        grow = vx.shape[0]  # grid rows
        gcol = vx.shape[1]  # grid cols
        ctrls = p.shape[0]  # control points

        # Precompute
        reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
        reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]

        w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha    # [ctrls, grow, gcol]
        w /= np.sum(w, axis=0, keepdims=True)                                               # [ctrls, grow, gcol]

        pstar = np.zeros((2, grow, gcol), np.float32)
        for i in range(ctrls):
            pstar += w[i] * reshaped_p[i]                                                   # [2, grow, gcol]

        phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
        phat = phat.reshape(ctrls, 2, 1, grow, gcol)                                        # [ctrls, 2, 1, grow, gcol]
        phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                                       # [ctrls, 1, 2, grow, gcol]
        reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
        pTwp = np.zeros((2, 2, grow, gcol), np.float32)
        for i in range(ctrls):
            pTwp += phat[i] * reshaped_w[i] * phat1[i]
        del phat1

        try:
            inv_pTwp = np.linalg.inv(pTwp.transpose(2, 3, 0, 1))                            # [grow, gcol, 2, 2]
            flag = False                
        except np.linalg.linalg.LinAlgError:                
            flag = True             
            det = np.linalg.det(pTwp.transpose(2, 3, 0, 1))                                 # [grow, gcol]
            det[det < 1e-8] = np.inf                
            reshaped_det = det.reshape(1, 1, grow, gcol)                                    # [1, 1, grow, gcol]
            adjoint = pTwp[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]                        # [2, 2, grow, gcol]
            adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]                  # [2, 2, grow, gcol]
            inv_pTwp = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                       # [grow, gcol, 2, 2]
        
        mul_left = reshaped_v - pstar                                                       # [2, grow, gcol]
        reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)        # [grow, gcol, 1, 2]
        mul_right = np.multiply(reshaped_w, phat, out=phat)                                 # [ctrls, 2, 1, grow, gcol]
        reshaped_mul_right = mul_right.transpose(0, 3, 4, 1, 2)                             # [ctrls, grow, gcol, 2, 1]
        out_A = mul_right.reshape(2, ctrls, grow, gcol, 1, 1)[0]                            # [ctrls, grow, gcol, 1, 1]
        A = np.matmul(np.matmul(reshaped_mul_left, inv_pTwp), reshaped_mul_right, out=out_A)    # [ctrls, grow, gcol, 1, 1]
        A = A.reshape(ctrls, 1, grow, gcol)                                                 # [ctrls, 1, grow, gcol]
        del mul_right, reshaped_mul_right, phat

        # Calculate q
        reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
        qstar = np.zeros((2, grow, gcol), np.float32)
        for i in range(ctrls):
            qstar += w[i] * reshaped_q[i]                                                   # [2, grow, gcol]
        del w, reshaped_w

        # Get final image transfomer -- 3-D array
        transformers = np.zeros((2, grow, gcol), np.float32)
        for i in range(ctrls):
            transformers += A[i] * (reshaped_q[i] - qstar)
        transformers += qstar
        del A

        # Correct the points where pTwp is singular
        if flag:
            blidx = det == np.inf    # bool index
            transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
            transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

        # Removed the points outside the border
        transformers[transformers < 0] = 0
        transformers[0][transformers[0] > grow - 1] = 0
        transformers[1][transformers[1] > gcol - 1] = 0

        return transformers.astype(np.int16)

    @staticmethod
    def _deform_mls(image, mask, src_points, dst_points, alpha):   
        height, width, _ = image.shape
        gridX = np.arange(width, dtype=np.int16)
        gridY = np.arange(height, dtype=np.int16)
        vy, vx = np.meshgrid(gridX, gridY, )

        affine1 = FeatMatcher._mls_affine_deformation(vy,vx,src_points,dst_points, alpha=alpha)
        aug1 = np.ones_like(image)
        aug1[vx, vy] = image[tuple(affine1)]
        aug1_mask = np.ones_like(mask)
        aug1_mask[vx, vy] = mask[tuple(affine1)]

        return aug1, aug1_mask
    
    @staticmethod
    def _add_matches_by_correspondance_score(src_pixels, dst_pixels, resize_factor=16):
        extra_src= []
        extra_dst= []
        for pix_i in range(len(src_pixels)):
            src_pix = src_pixels[pix_i]
            dst_pix = dst_pixels[pix_i]
            score = src_pix[-1]
            for i in range(0, resize_factor, resize_factor//2):
                for j in range(0, resize_factor, resize_factor//2):
                    r = np.random.uniform()
                    if r < score:
                        extra_src.append(src_pix[:-1] * resize_factor + (i,j))     
                        extra_dst.append(dst_pix[:-1] * resize_factor + (i,j))     
        
        src_pixels= np.stack(extra_src,0)
        dst_pixels= np.stack(extra_dst,0)
        return src_pixels, dst_pixels

    @staticmethod
    def _deform_image(src_image, src_mask, src_pixels, dst_pixels, alpha=0.1):
        src_pixels, dst_pixels = FeatMatcher._add_matches_by_correspondance_score(src_pixels, dst_pixels)
        transformed_image, transformed_mask = FeatMatcher._deform_mls(src_image, 
                                                        src_mask, 
                                                        src_pixels, 
                                                        dst_pixels, 
                                                        alpha=alpha)
        transformed_image[transformed_mask == 0] = 0
        transformed_mask = np.tile(transformed_mask[...,None], 3)

        return transformed_image, transformed_mask

    @staticmethod    
    def _resize_mask(mask : np.array, dim : int):
        resized_mask = PIL.Image.fromarray(mask).resize((dim, dim), resample=PIL.Image.LANCZOS)
        resized_mask = pil_to_numpy(resized_mask.convert("L"))
        resized_mask = resized_mask >= 128 
        reshaped_mask = resized_mask.reshape([dim * dim])

        return reshaped_mask


class FeatMatchDebuger(FeatMatcher):
    def __init__(self, device, prev_pipeline=None) -> None:
        super().__init__(device, prev_pipeline)

    def compute_matching(self,
                         ref_mask : PIL.Image.Image, 
                         target_mask : PIL.Image.Image, 
                         ref_img_base : PIL.Image.Image = None, 
                         target_img_base : PIL.Image.Image = None,
                         return_debug_images : bool = False, 
                         ):
        if isinstance(ref_img_base, PIL.Image.Image):
            ref_img_base = pil_to_numpy(ref_img_base.convert("RGB"))
        if isinstance(target_img_base, PIL.Image.Image):
            target_img_base = pil_to_numpy(target_img_base.convert("RGB"))
        if isinstance(ref_mask, PIL.Image.Image):
            ref_mask = pil_to_numpy(ref_mask.convert("L"))
        if isinstance(target_mask, PIL.Image.Image):
            target_mask = pil_to_numpy(target_mask.convert("L"))

        ref_mask[ref_mask<125] = 0
        ref_mask[ref_mask>=125] = 255
        target_mask[target_mask<125] = 0
        target_mask[target_mask>=125] = 255

        assert return_debug_images is None or \
            (return_debug_images is not None and ref_img_base is not None and target_img_base is not None)
        
        
        batch, channel, width, height = self.features.shape

        ref_feat, target_feat = self.features.chunk(batch)

        reshaped_ref_mask = FeatMatchDebuger._resize_mask(ref_mask, width)
        reshaped_target_mask = FeatMatchDebuger._resize_mask(target_mask, width)
        
        matching_pixels, matching_matrix = FeatMatchDebuger._mnn_matcher(ref_feat, target_feat, reshaped_ref_mask, reshaped_target_mask)
        
        src_pixels, dst_pixels = FeatMatchDebuger._filter_matches_arrays(matching_pixels)
        deformed_ref_image, deformed_ref_mask = FeatMatchDebuger._deform_image(ref_img_base, ref_mask, src_pixels, dst_pixels)
        ref_on_target = np.where(deformed_ref_mask * np.tile(target_mask[...,None], 3) == 1, deformed_ref_image, target_img_base)

        match_drawing_image = FeatMatchDebuger.draw_matching(ref_img_base, target_img_base, src_pixels, dst_pixels)
        
        ref_on_target = Image.fromarray(ref_on_target)
        self._clear_features()

        return {
                "x_g_p": ref_on_target,
                "deformed_mask": deformed_ref_mask,
                "visualization": match_drawing_image
                }

    @staticmethod
    def draw_matching(img1, img2, src_pixels, dst_pixels, resize_factor=16):
        if isinstance(img1, PIL.Image.Image):
            img1 = pil_to_numpy(img1.convert("RGB"))
        if isinstance(img2, PIL.Image.Image):
            img2 = pil_to_numpy(img2.convert("RGB"))

        res= np.concatenate((img1, img2), 1)
        
        for i, idx in enumerate((range(src_pixels.shape[0]))):
            src_x, src_y, src_score = src_pixels[idx] * (resize_factor, resize_factor, 1) 
            dst_x, dst_y, dst_score = dst_pixels[idx] * (resize_factor, resize_factor, 1) 
            
            dst_y += img1.shape[0]
            if src_x == 0 and src_y == 0: continue
            if src_score <= 0 : 
                continue

            width = int(src_score * 8) 
            src_x = int(src_x)
            src_y = int(src_y)
            dst_x = int(dst_x)
            dst_y = int(dst_y)
            
            color = FeatMatchDebuger._random_color()
            res[src_x-width:src_x+width, src_y-width:src_y+width] = color
            res[dst_x-width:dst_x+width, dst_y-width:dst_y+width] = color
            cv2.line(res, (int(src_y), int(src_x)), (int(dst_y), int(dst_x)), color, 2)  

        return res
    
    @staticmethod
    def _random_color():
        # Generate three random integers in the range 0 to 255
        blue = np.random.randint(0, 256)
        green = np.random.randint(0, 256)
        red = np.random.randint(0, 256)

        # Create a tuple representing the random color in BGR format
        color = (blue, green, red)

        return color
    


    def get_feature_visualization(  self,
                                    ref_mask : PIL.Image.Image, 
                                    target_mask : PIL.Image.Image, 
                                    ref_img_base : PIL.Image.Image = None, 
                                    target_img_base : PIL.Image.Image = None,
                                  ):
        
        ref_img_base = to_array(ref_img_base.convert("RGB"))
        target_img_base = to_array(target_img_base.convert("RGB"))
        ref_mask = to_array(ref_mask.convert("L"))
        target_mask = to_array(target_mask.convert("L"))

        ref_mask[ref_mask<125] = 0
        ref_mask[ref_mask>=125] = 255
        target_mask[target_mask<125] = 0
        target_mask[target_mask>=125] = 255

        batch, channel, width, height = self.features.shape

        ref_feat, target_feat = self.features.chunk(batch)
        reshaped_ref_mask = FeatMatchDebuger._resize_mask(ref_mask, width)
        reshaped_target_mask = FeatMatchDebuger._resize_mask(target_mask, width)

        matching_pixels, matching_matrix = FeatMatchDebuger._mnn_matcher(ref_feat, target_feat, reshaped_ref_mask, reshaped_target_mask)
        matching_mat_vis = matching_matrix.reshape(32, 32, 32, 32).mean(axis=(1, 3))
        matching_mat_vis = (matching_mat_vis - matching_mat_vis.min()) / (matching_mat_vis.max() - matching_mat_vis.min())
        matching_mat_vis = cv2.applyColorMap(pil_to_numpy(matching_mat_vis.squeeze()*255, dtype=np.uint8), cv2.COLORMAP_HOT)
        matching_mat_vis = cv2.resize(matching_mat_vis, (512,512), interpolation=cv2.INTER_NEAREST)
        matching_mat_vis = cv2.cvtColor(matching_mat_vis, cv2.COLOR_RGB2BGR)
        
        ref_feat = ref_feat.sum(1) / ref_feat.shape[1]
        target_feat = target_feat.sum(1) / target_feat.shape[1]
        feat_min = min(ref_feat.min(), target_feat.min())
        feat_max = max(ref_feat.max(), target_feat.max())
        
        ref_feat = (ref_feat - feat_min) / (feat_max - feat_min)
        # ref_feat = (ref_feat - ref_feat.min()) / (ref_feat.max() - ref_feat.min())
        reshaped_ref_mask = reshaped_ref_mask.reshape(32,32)
        reshaped_ref_mask = reshaped_ref_mask > 0.5
        ref_feat= ref_feat.cpu()*reshaped_ref_mask

        # target_feat = (target_feat - target_feat.min()) / (target_feat.max() - target_feat.min())
        target_feat = (target_feat - feat_min) / (feat_max - feat_min)
        reshaped_target_mask = reshaped_target_mask > 0.5
        reshaped_target_mask = reshaped_target_mask.reshape(32,32)
        target_feat= target_feat.cpu()*reshaped_target_mask

        color_wheel_image_ref = cv2.applyColorMap(pil_to_numpy(ref_feat.squeeze()*255, dtype=np.uint8), cv2.COLORMAP_TWILIGHT_SHIFTED)
        color_wheel_image_ref = cv2.resize(color_wheel_image_ref, (512,512), interpolation=cv2.INTER_NEAREST)

        color_wheel_image_target = cv2.applyColorMap(pil_to_numpy(target_feat.squeeze()*255, dtype=np.uint8), cv2.COLORMAP_TWILIGHT_SHIFTED)
        color_wheel_image_target = cv2.resize(color_wheel_image_target, (512,512), interpolation=cv2.INTER_NEAREST)

        return color_wheel_image_ref, color_wheel_image_target, matching_mat_vis
