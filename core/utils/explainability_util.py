import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision.utils as vutils
import cv2

# Directory where to save the blending weights log
blending_log_dir = "blending_weights_log"
os.makedirs(blending_log_dir, exist_ok=True)

# File to store blending weights over iterations
log_file = os.path.join(blending_log_dir, "blending_weights.txt")



@torch.no_grad()
def visualize_counterfactual_pose_effect(model, data, joint_idx=17, axis=0, delta=0.3, width=None, height=None, ray_mask=None, save_path="counterfactuals"):
    os.makedirs(save_path, exist_ok=True)

    device = next(model.parameters()).device
    pose_orig = data['dst_posevec'].clone().to(device)
    pose_pert = pose_orig.clone()

    # Perturb joint
    if pose_pert.dim() == 2:
        pose_pert[0, joint_idx * 3 + axis] += delta
    else:
        pose_pert[joint_idx * 3 + axis] += delta

    # Prepare data
    data_orig = data.copy()
    data_pert = data.copy()
    data_orig['dst_posevec'] = pose_orig
    data_pert['dst_posevec'] = pose_pert

    iter_val = data.get('iter_val', torch.tensor([1e7], device=device))
    rendered_orig = model(**data_orig, iter_val=iter_val)['rgb']  # [N, 3]
    rendered_pert = model(**data_pert, iter_val=iter_val)['rgb']  # [N, 3]

    # Calculate L1 diff
    diff = torch.abs(rendered_orig - rendered_pert).mean(dim=-1)  # [N]
    diff_np = diff.cpu().numpy()
    diff_norm = (diff_np - diff_np.min()) / (diff_np.max() - diff_np.min() + 1e-8)

    # Apply colormap
    heatmap = cm.inferno(diff_norm)[:, :3]  # Drop alpha, shape [N, 3]
    heat_rgb = (heatmap * 255).astype(np.uint8)

    # Prepare background image shape
    bg_color = np.array([0.0, 0.0, 0.0])
    height = height or data['img_height']
    width = width or data['img_width']

    def to_img(tensor):
        img = np.full((height * width, 3), bg_color, dtype='float32')
        img[ray_mask] = tensor.detach().cpu().numpy()
        return img.reshape(height, width, 3)

    img_orig = to_img(rendered_orig)
    img_pert = to_img(rendered_pert)

    # Reshape heatmap
    heat_rgb_img = np.full((height * width, 3), 0, dtype=np.uint8)
    heat_rgb_img[ray_mask] = heat_rgb
    heat_rgb_img = heat_rgb_img.reshape(height, width, 3)

    # Overlay heatmap
    overlay = cv2.addWeighted((img_orig * 255).astype(np.uint8), 0.7, heat_rgb_img, 0.3, 0)

    # Save side-by-side comparison
    stacked = np.concatenate([img_orig, img_pert, heat_rgb_img], axis=1)
    output_img = (stacked * 255).clip(0, 255).astype(np.uint8)

    save_base = f'cf_joint{joint_idx}_axis{axis}_delta{delta}'
    Image.fromarray(output_img).save(os.path.join(save_path, f'{save_base}.png'))
    Image.fromarray(overlay).save(os.path.join(save_path, f'{save_base}_overlay.png'))

    print(f"[✓] Saved counterfactual: {save_base}.png and overlay")
def log_blending_weight(blending_weight: torch.Tensor, iteration: int):
    """Logs the blending weight to a text file."""
    weight = torch.sigmoid(blending_weight).item()
    with open(log_file, "a") as f:
        f.write(f"{iteration},{weight:.6f}\n")

def plot_blending_weights(log_file_path=log_file, save_path=None):
    """Plots the blending weights from log file over iterations."""
    iterations = []
    weights = []

    with open(log_file_path, "r") as f:
        for line in f:
            iter_val, w = line.strip().split(",")
            iterations.append(int(iter_val))
            weights.append(float(w))

    plt.figure(figsize=(8, 4))
    plt.plot(iterations, weights, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Blending Weight (σ_r vs σ_t)")
    plt.title("Adaptive Blending Weight over Iterations")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
