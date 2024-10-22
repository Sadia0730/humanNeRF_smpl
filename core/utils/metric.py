import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import torch
def calculate_metrics(true_img, pred_img):
    global lpips_model  # Declare the use of the global variable

    # Check if the model is already initialized
    if 'lpips_vgg' not in globals():
        from third_parties.lpips import LPIPS
        lpips_model = LPIPS(net='vgg').cuda()  # Initialize if not present

    # Assuming images are in range [0, 255]
    pred_img = (pred_img / 255.0).astype('float32')
    true_img = (true_img / 255.0).astype('float32')
    print("True image shape:", true_img.shape)
    print("Predicted image shape:", pred_img.shape)
    print("True image min/max:", true_img.min(), true_img.max())
    print("Predicted image min/max:", pred_img.min(), pred_img.max())
    print("True image data type:", true_img.dtype)
    print("Predicted image data type:", pred_img.dtype)

    # Convert images to tensor and add batch dimension
    true_tensor = torch.from_numpy(true_img.transpose(2, 0, 1)).unsqueeze(0).cuda()
    pred_tensor = torch.from_numpy(pred_img.transpose(2, 0, 1)).unsqueeze(0).cuda()

    psnr_val = psnr(true_img, pred_img, data_range=1)
    ssim_val = ssim(true_img, pred_img, data_range=1, multichannel=True, win_size=7, channel_axis=-1)
    mse_val = np.mean((true_img - pred_img) ** 2)
    mae_val = np.mean(np.abs(true_img - pred_img))

    # Calculate LPIPS
    lpips_val = lpips_model(true_tensor, pred_tensor).item()

    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'MSE': mse_val,
        'MAE': mae_val,
        'LPIPS': lpips_val
    }
