import os

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from third_parties.lpips import LPIPS
from torchvision.models import vgg16
from core.utils.network_util import set_requires_grad
from core.train import create_lr_updater
from core.data import create_dataloader
from core.utils.train_util import cpu_data_to_gpu, Timer
from core.utils.image_util import tile_images, to_8b_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from configs import cfg

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']


def visualize_data_with_bbox(img, bbox, title="Image with Bounding Box"):
    img = img.cpu()
    fig, ax = plt.subplots()
    ax.imshow(img)
    rect = patches.Rectangle((bbox['min_xyz'][0], bbox['min_xyz'][1]),
                             bbox['max_xyz'][0] - bbox['min_xyz'][0],
                             bbox['max_xyz'][1] - bbox['min_xyz'][1],
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.title(title)
    plt.show()
    

def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch
    print(f"rgbs shape: {rgbs.shape}")
    print(f"targets shape: {targets.shape}")
    print(f"div_indices shape: {div_indices.shape}")
    print(f"div_indices: {div_indices}")
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

        self.optimizer = optimizer
        self.update_lr = create_lr_updater()
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

    def get_img_rebuild_loss(self, loss_names, rgb, target):
        losses = {}

        if "mse" in loss_names:
            losses["mse"] = img2mse(rgb, target)

        if "l1" in loss_names:
            losses["l1"] = img2l1(rgb, target)

        if "lpips" in loss_names:
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                                    scale_for_lpips(target.permute(0, 3, 1, 2)))
            losses["lpips"] = torch.mean(lpips_loss)

        return losses
    def calculate_feature_loss(self,net_output,data):
            # Extract and process patches
            generated_patches = net_output['rgb']  
            real_patches = data['target_patches']  
            print(f"generated_patches {generated_patches.shape}")
            print(f"real_patches {real_patches.shape}")
            generated_patches = _unpack_imgs(generated_patches, data['patch_masks'], data['bgcolor'] / 255.,
                                     real_patches, data['patch_div_indices'])
            print(f"generated_patches {generated_patches.shape}")
            print(f"real_patches {real_patches.shape}")
            # Initialize patch-based feature loss
            patch_feature_loss = 0

            # Loop over each patch to compute feature-based loss
            for idx in range(real_patches.shape[0]): 
                print(f"real_patches idx{real_patches[idx].shape}")
                print(f"generated_patches {generated_patches[idx].shape}")
                real_patch = real_patches[idx].permute(2, 0, 1).unsqueeze(0)  # Permute and add batch dimension
                generated_patch = generated_patches[idx].permute(2, 0, 1).unsqueeze(0)  # Permute and add batch dimension

                # Extract features for each patch
                with torch.no_grad():
                    real_features = self.feature_extractor(real_patch)
                    print(f"Real Feature Shape: {real_features.shape}")
                    generated_features = self.feature_extractor(generated_patch)
                    print(f"generated Feature Shape: {generated_features.shape}")
                # Calculate and accumulate feature-based loss for each patch
                patch_feature_loss += F.mse_loss(generated_features, real_features)
            # Average the feature loss across all patches
            patch_feature_loss /= real_patches.shape[0]
            return patch_feature_loss

    def get_loss(self, net_output, 
                 patch_masks, bgcolor, targets, div_indices):

        lossweights = cfg.train.lossweights
        loss_names = list(lossweights.keys())

        rgb = net_output['rgb']
        losses = self.get_img_rebuild_loss(
                        loss_names, 
                        _unpack_imgs(rgb, patch_masks, bgcolor,
                                     targets, div_indices), 
                        targets)

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
        for batch_idx, batch in enumerate(train_dataloader):
            if self.iter > cfg.train.maxiter:
                break
            print(f"frame_name in train: {batch['frame_name']}")
            self.optimizer.zero_grad()

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            net_output = self.network(**data)



            #print(f"batch_idx: {batch_idx} net_output:{net_output}")
            # width = batch['img_width']
            # height = batch['img_height']
            # ray_mask = batch['ray_mask']
            # torch.set_printoptions(threshold=torch.numel(ray_mask))
            # print(f"ray_mask {ray_mask.shape}")
            # print(f"ray_mask {ray_mask}")
            # truth = np.full(
            #             (height * width, 3), np.array(cfg.bgcolor)/255., 
            #             dtype='float32')
            # target_rgbs = batch['target_rgbs']
            # print(f"target_rgbs: {target_rgbs.shape}") #2095,3
            # print(f"target_rgbs: {target_rgbs}")
            # truth[ray_mask] = target_rgbs
            # print(f"Truth data: {truth}")
            # print(f"Truth: {truth.shape}")
            # truth = to_8b_image(truth.reshape((height, width, -1)))
            # print(f"After Truth: {truth.shape}")
            # Image.fromarray(tiled_image).save(
            # os.path.join(cfg.logdir, "prog_{:06}.jpg".format(self.iter)))
            
            # patch_feature_loss = self.calculate_feature_loss(net_output,data)
            train_loss, loss_dict = self.get_loss(
                net_output=net_output,
                patch_masks=data['patch_masks'],
                bgcolor=data['bgcolor'] / 255.,
                targets=data['target_patches'],
                div_indices=data['patch_div_indices'])
           
            train_loss.backward()
            self.optimizer.step()

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

                self.update_lr(self.optimizer, self.iter)

                self.iter += 1
    
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

        print('Evaluate Progress Images ...')

        images = []
        is_empty_img = False
        for _, batch in enumerate(tqdm(self.prog_dataloader)):
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
            # print(f"target_rgbs: {target_rgbs.shape}")
            # print(f"progress_rgb:{rgb.shape}")
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
            # if self.iter <= 5000 and \
            #     np.allclose(rendered, np.array(cfg.bgcolor), atol=5.):
            #     is_empty_img = True
            #     break

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
        self.optimizer.load_state_dict(ckpt['optimizer'])
