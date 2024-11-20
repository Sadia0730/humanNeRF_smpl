import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from third_parties.lpips import LPIPS
from torchvision.models import vgg16
from core.utils.network_util import set_requires_grad, check_for_nans
from core.train import create_lr_updater
from core.data import create_dataloader
from core.utils.train_util import cpu_data_to_gpu, Timer
from core.utils.image_util import tile_images, to_8b_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from configs import cfg
from piq import SSIMLoss
from core.utils.metric import calculate_metrics

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']

# def reconstruct_image_from_patches(patch_rgb_image, patch_info_xy_min, patch_info_xy_max, patch_masks, H, W):
#     # Initialize the reconstructed image and a weight mask
#     C = patch_rgb_image.shape[-1]  # Number of color channels
#     print(f"C: {C}")
#     print(f"patch_info_xy_min shape: {patch_info_xy_min.shape}")
#     print(f"patch_info_xy_max shape: {patch_info_xy_max.shape}")
#     print(f"patch_masks: {patch_masks.shape}")
#     print(f"patch_info_xy_min checking: {patch_info_xy_min} patch_info_xy_max checking: {patch_info_xy_max}")

#     # Initialize the reconstructed image and a weight mask
#     reconstructed_image = np.zeros((H, W, C))  # For RGB images, C=3.
#     coverage_map = np.zeros((H, W))  # Tracks how many patches contribute to each pixel.

#     # Loop through each patch and its corresponding info
#     print(f"patch_rgb_image type: {type(patch_rgb_image)}")
#     print(f"patch_rgb_image shape: {patch_rgb_image.shape if hasattr(patch_rgb_image, 'shape') else 'Unknown'}")

#     for patch, xy_min, xy_max in zip(patch_rgb_image, patch_info_xy_min, patch_info_xy_max):
#         # Convert PyTorch tensors to NumPy integers
#         x_min, y_min = xy_min.cpu().numpy().astype(int)
#         x_max, y_max = xy_max.cpu().numpy().astype(int)
        
#         # Clip indices to ensure they are within bounds
#         x_min = max(0, x_min)
#         x_max = min(W, x_max)
#         y_min = max(0, y_min)
#         y_max = min(H, y_max)
#         # Convert patch to NumPy
#         patch_np = patch.detach().cpu().numpy()
#         # Check for size consistency
#         assert patch.shape == (y_max - y_min, x_max - x_min, C), \
#             f"Patch size {patch.shape} does not match bounding box size {(y_max - y_min, x_max - x_min, C)}"

#         # Update the reconstructed image and coverage map
#         reconstructed_image[y_min:y_max, x_min:x_max] += patch_np
#         coverage_map[y_min:y_max, x_min:x_max] += 1  # Update coverage map


#         # Debugging information
#         # print(f"Patch : xy_min=({x_min}, {y_min}), xy_max=({x_max}, {y_max})")
#         # print(f"Patch shape: {patches[i].shape}, Mask sum: {np.sum(patch_info['mask'][i])}")

#             # Print min and max values before normalization
#     print(f"Before Normalization: Min = {reconstructed_image.min()}, Max = {reconstructed_image.max()}")

#     # Save the reconstructed image before normalization
#     reconstructed_image_uint8 = (np.clip(reconstructed_image, 0, 1) * 255).astype(np.uint8)
#     cv2.imwrite("reconstructed_image_before_normalization.png", cv2.cvtColor(reconstructed_image_uint8, cv2.COLOR_RGB2BGR))

#     print(f"After Normalization: Min = {reconstructed_image.min()}, Max = {reconstructed_image.max()}")

#    # Save the reconstructed image after normalization
#     reconstructed_image_uint8 = (np.clip(reconstructed_image, 0, 1) * 255).astype(np.uint8)
#     cv2.imwrite("reconstructed_image_after_normalization.png", cv2.cvtColor(reconstructed_image_uint8, cv2.COLOR_RGB2BGR))

#     return reconstructed_image
def reconstruct_image_from_patches(patch_rgb_image, patch_info_xy_min, patch_info_xy_max, patch_masks, H, W):
    C = patch_rgb_image.shape[-1]
    reconstructed_image = np.zeros((H, W, C))
    coverage_map = np.zeros((H, W))
    
    # Debug total number of patches and masks
    n_patches = len(patch_rgb_image)
    print(f"Total number of patches: {n_patches}")
    print(f"Patch masks shape: {patch_masks.shape if patch_masks is not None else 'No masks'}")
    
    # Create debug visualization
    debug_viz = np.zeros((H, W, 3), dtype=np.uint8)
    
    for i, (patch, xy_min, xy_max) in enumerate(zip(patch_rgb_image, patch_info_xy_min, patch_info_xy_max)):
        # Convert coordinates
        x_min, y_min = xy_min.cpu().numpy().astype(int)
        x_max, y_max = xy_max.cpu().numpy().astype(int)
        
        # Get mask for this patch if available
        mask = None
        if patch_masks is not None:
            mask = patch_masks[i].cpu().numpy().astype(bool)
        
        # Clip boundaries
        x_min, x_max = max(0, x_min), min(W, x_max)
        y_min, y_max = max(0, y_min), min(H, y_max)
        
        # Skip invalid patches
        if x_min >= x_max or y_min >= y_max:
            print(f"Skipping invalid patch {i}: coords ({x_min},{y_min}) to ({x_max},{y_max})")
            continue
        
        patch_np = patch.detach().cpu().numpy()
        
        # Apply mask if available
        if mask is not None:
            # Ensure mask and patch have compatible shapes
            mask = mask[:y_max-y_min, :x_max-x_min]
            patch_np = patch_np * mask[..., None]  # Broadcast mask to all channels
        
        try:
            # Update reconstruction with masked patch
            patch_region = reconstructed_image[y_min:y_max, x_min:x_max]
            patch_coverage = coverage_map[y_min:y_max, x_min:x_max]
            
            valid_mask = np.ones_like(patch_np[..., 0], dtype=bool)
            if mask is not None:
                valid_mask = mask
            
            # Only update pixels where the patch has valid data
            patch_region[valid_mask] += patch_np[valid_mask]
            patch_coverage[valid_mask] += 1
            
            # Add to debug visualization
            color = np.array([hash(str(i)) % 256, hash(str(i+1)) % 256, hash(str(i+2)) % 256])
            debug_viz[y_min:y_max, x_min:x_max][valid_mask] = color
            
        except ValueError as e:
            print(f"Error processing patch {i}: {str(e)}")
            print(f"Patch shape: {patch_np.shape}")
            print(f"Region shape: {(y_max-y_min, x_max-x_min, C)}")
    
    # Save debug visualization
    cv2.imwrite("patch_placement_debug.png", debug_viz)
    
    # Print coverage analysis
    covered_pixels = np.count_nonzero(coverage_map)
    total_pixels = H * W
    print(f"\nCoverage Analysis:")
    print(f"Covered pixels: {covered_pixels}/{total_pixels} ({covered_pixels/total_pixels*100:.2f}%)")
    print(f"Max overlap: {int(coverage_map.max())} patches")
    
    # Normalize with careful handling of divisions
    mask = coverage_map > 0
    reconstructed_image[mask] /= coverage_map[mask, None]
    
    # Clip and convert to uint8
    reconstructed_image_uint8 = (np.clip(reconstructed_image, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(f"reconstructed_final_{100}.png", cv2.cvtColor(reconstructed_image_uint8, cv2.COLOR_RGB2BGR))
    
    return reconstructed_image

def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    print(f"rgbs shape: {rgbs.shape}")
    print(f"targets shape: {targets.shape}")
    print(f"div_indices shape: {div_indices.shape}")
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch
    # print(f"div_indices: {div_indices}")
    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]
    print(f"patch_img shape: {patch_imgs.shape}")
    return patch_imgs


def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


class Trainer(object):
    def __init__(self, network, optimizer):
        print('\n********** Init Trainer ***********')

        network = network.cuda().deploy_mlps_to_secondary_gpus()
        self.network = network

        # self.optimizer = optimizer
        self.optimizer, self.scheduler = optimizer 
        # self.update_lr = create_lr_updater()
        self.feature_extractor = vgg16(pretrained=True).features[:10].cuda()  
        set_requires_grad(self.feature_extractor, requires_grad=False)  


        if cfg.resume and Trainer.ckpt_exists(cfg.load_net):
            self.load_ckpt(f'{cfg.load_net}')
        else:
            self.iter = 0
            self.save_ckpt('init')
            self.iter = 1

        self.timer = Timer()

        if "lpips" in cfg.train.lossweights.keys():
            self.lpips = LPIPS(net='vgg')
            set_requires_grad(self.lpips, requires_grad=False)
            self.lpips = nn.DataParallel(self.lpips).cuda()

        print("Load Progress Dataset ...")
        self.prog_dataloader = create_dataloader(data_type='progress')

        print('************************************')

    @staticmethod
    def get_ckpt_path(name):
        return os.path.join(cfg.logdir, f'{name}.tar')

    @staticmethod
    def ckpt_exists(name):
        return os.path.exists(Trainer.get_ckpt_path(name))

    ######################################################3
    ## Training

    def calculate_segmentation_loss(self, patch_rgb, target_alpha, H, W):
        # Convert patch_rgb to grayscale using the luminance formula
        patch_grayscale = 0.2989 * patch_rgb[..., 0] + 0.5870 * patch_rgb[..., 1] + 0.1140 * patch_rgb[..., 2]
        print(f"patch_rgb: {patch_rgb.shape}")
        print(f"patch_grayscale: {patch_grayscale.shape}")
        print(f"target_alpha: {target_alpha.shape}")
        # Reshape the target_alpha if it has 3 channels (convert to single channel)
        target_alpha = target_alpha.mean(dim=-1) if target_alpha.shape[-1] == 3 else target_alpha
        print(f"target_alpha: {target_alpha.shape}")
        # Threshold the grayscale patch to create a binary mask
        pred_mask = (patch_grayscale > 0.5).float()  # Binary mask from predicted RGB


        # Calculate a binary cross-entropy or MSE loss between the predicted and target masks
        segmentation_loss = F.mse_loss(pred_mask, target_alpha)

        return segmentation_loss

    def get_img_rebuild_loss(self, loss_names, rgb, target, alpha_patches,H,W):
        losses = {}

        if "mse" in loss_names:
            losses["mse"] = img2mse(rgb, target)

        if "l1" in loss_names:
            losses["l1"] = img2l1(rgb, target)

        if "lpips" in loss_names:
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                                    scale_for_lpips(target.permute(0, 3, 1, 2)))
            losses["lpips"] = torch.mean(lpips_loss)

        if "silhouette" in loss_names:
            silhouette_loss = self.calculate_segmentation_loss(rgb, alpha_patches, H, W)
            losses["silhouette"] = silhouette_loss
            print(f"iter in constr: {self.iter}")
        # # Add SSIM loss calculation
        # if "ssim" in loss_names:
        #     print(f"RGB Min: {rgb.min()}, Max: {rgb.max()}")
        #     print(f"Target Min: {target.min()}, Max: {target.max()}")
        #     print(rgb.shape)
        #     print(target.shape)
        #     ssim_loss_fn = SSIMLoss(data_range=1.0)  # Assuming normalized data in [0, 1]
        #     ssim_loss = ssim_loss_fn(torch.clamp(rgb,0,1).permute(0, 3, 1, 2), torch.clamp(target,0,1).permute(0, 3, 1, 2))
        #     losses["ssim"] = 1 - ssim_loss

        return losses

    def get_loss(self, net_output, 
                 patch_masks, bgcolor, targets, div_indices,alpha, alpha_patches, ray_mask,  patch_info_xy_min, patch_info_xy_max):
        H, W = alpha.shape[:2]
        lossweights = cfg.train.lossweights
        loss_names = list(lossweights.keys())
        rgb = net_output['rgb']
        patch_rgb_image = _unpack_imgs(rgb, patch_masks, bgcolor,
                                     targets, div_indices)

        print(f"Alpha Shape: {alpha.shape}")
        print(f"rgb Shape in train: {rgb.shape}")
        print(f"patch_rgb_image Shape in train: {patch_rgb_image.shape}")
        print(f"patch_masks Shape: {patch_masks.shape}")
        print(f"ray_mask Shape: {ray_mask.shape}")
        print(f"targets Shape: {targets.shape}")
        print(f"alpha_patches Shape: {alpha_patches.shape}")
     


        losses = self.get_img_rebuild_loss(
                        loss_names, 
                        _unpack_imgs(rgb, patch_masks, bgcolor,
                                     targets, div_indices), 
                        targets, alpha_patches,H,W)

        train_losses = [
            weight * losses[k] for k, weight in lossweights.items()
        ]

        return sum(train_losses), \
               {loss_names[i]: train_losses[i] for i in range(len(loss_names))}

    def train_begin(self, train_dataloader):
        assert train_dataloader.batch_size == 1

        self.network.train()
        cfg.perturb = cfg.train.perturb

    def train_end(self):
        pass

    def train(self, epoch, train_dataloader):
        self.train_begin(train_dataloader=train_dataloader)

        self.timer.begin()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            if self.iter > cfg.train.maxiter:
                break
            self.optimizer.zero_grad()
            # only access the first batch as we process one image one time
            print(batch.items())
            for k, v in batch.items():
                batch[k] = v[0]

            batch['iter_val'] = torch.full((1,), self.iter)
            ray_mask = batch['ray_mask']
            data = cpu_data_to_gpu(
                batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            net_output = self.network(**data)

            with torch.autograd.detect_anomaly():
                train_loss, loss_dict = self.get_loss(
                    net_output=net_output,
                    patch_masks=data['patch_masks'],
                    bgcolor=data['bgcolor'] / 255.,
                    targets=data['target_patches'],
                    div_indices=data['patch_div_indices'],
                    alpha=data['alpha'],
                    alpha_patches=data['alpha_patches'],
                    ray_mask=ray_mask,
                    patch_info_xy_min=data['patch_info_xy_min'],
                    patch_info_xy_max=data['patch_info_xy_max']
                    )
                for loss_name, loss_val in loss_dict.items():
                    if check_for_nans(f"{loss_name} loss", loss_val):
                        print(f"NaN detected in {loss_name} loss")

                if torch.isnan(train_loss):
                    print("NaN detected in train_loss, proceeding with debug")


                train_loss.backward()
                for name, param in self.network.named_parameters():
                    if param.grad is not None and check_for_nans(f"{name} grad", param.grad):
                        print(f"NaN detected in gradient of {name}")
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                # Additional gradient clipping only for density_mlp
                torch.nn.utils.clip_grad_norm_(self.network.density_mlp.parameters(), max_norm=1.0)

                self.optimizer.step()
                epoch_loss += train_loss.item()
            if self.iter % cfg.train.log_interval == 0:
                loss_str = f"Loss: {train_loss.item():.4f} ["
                for k, v in loss_dict.items():
                    loss_str += f"{k}: {v.item():.4f} "
                loss_str += "]"

                log_str = 'Epoch: {} [Iter {}, {}/{} ({:.0f}%), {}] {}'
                log_str = log_str.format(
                    epoch, self.iter,
                    batch_idx * cfg.train.batch_size, len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), 
                    self.timer.log(),
                    loss_str)
                print(log_str)

            is_reload_model = False
            if self.iter in [100, 300, 1000, 2500] or \
                self.iter % cfg.progress.dump_interval == 0:
                is_reload_model = self.progress()

            if not is_reload_model:
                if self.iter % cfg.train.save_checkpt_interval == 0:
                    self.save_ckpt('latest')

                if cfg.save_all:
                    if self.iter % cfg.train.save_model_interval == 0:
                        self.save_ckpt(f'iter_{self.iter}')

                # self.update_lr(self.optimizer, self.iter)
                print(f"Iter: {self.iter}r")
                self.iter += 1
                
        epoch_loss = float(epoch_loss / len(train_dataloader))  # Convert to scalar Python float
        # Call scheduler once per epoch
        self.scheduler.step(epoch_loss)
    
    def finalize(self):
        self.save_ckpt('latest')

    ######################################################3
    ## Progress

    def progress_begin(self):
        self.network.eval()
        cfg.perturb = 0.

    def progress_end(self):
        self.network.train()
        cfg.perturb = cfg.train.perturb

    def progress(self):
        self.progress_begin()
        metrics_data = []
        print('Evaluate Progress Images ...')

        images = []
        is_empty_img = False
        for idx, batch in enumerate(tqdm(self.prog_dataloader)):
            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rendered = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                    batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = self.network(**data)

            rgb = net_output['rgb'].data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs']



            print(f"target_rgbs: {target_rgbs.shape}")
            print(f"progress_rgb:{rgb.shape}")
            # print(f"ray_mask Shape:{ray_mask.shape}")
            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs
            # print(f" count of ray mask : {torch.sum(ray_mask)}")
            # print(f"Rendered: {rendered.shape}")
            truth = to_8b_image(truth.reshape((height, width, -1)))
            rendered = to_8b_image(rendered.reshape((height, width, -1)))
            # print(f"After Truth: {truth.shape}")
            # print(f"Rendered: {rendered.shape}")
            images.append(np.concatenate([truth, rendered], axis=1))

            # check if we create empty images (only at the begining of training)
            if self.iter <= 5000 and \
                np.allclose(rendered, np.array(cfg.bgcolor), atol=5.):
                is_empty_img = True
                break

        tiled_image = tile_images(images)
        print("in progress:")
        Image.fromarray(tiled_image).save(
            os.path.join(cfg.logdir, "prog_{:06}.jpg".format(self.iter)))
        print("in progress saved:")
        if is_empty_img:
            print("Produce empty images; reload the init model.")
            self.load_ckpt('init')
            
        self.progress_end()
        return is_empty_img


    ######################################################3
    ## Utils

    def save_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Save checkpoint to {path} ...")

        torch.save({
            'iter': self.iter,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Load checkpoint from {path} ...")
        
        ckpt = torch.load(path, map_location='cuda:0')
        self.iter = ckpt['iter'] + 1

        self.network.load_state_dict(ckpt['network'], strict=False)
        self.network = self.network.cuda()  # or self.network.to('cuda:0') for consistency
        for submodule in self.network.children():
            submodule = submodule.cuda()
        self.optimizer.load_state_dict(ckpt['optimizer'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()  # Ensure all optimizer tensors are moved to the GPU

