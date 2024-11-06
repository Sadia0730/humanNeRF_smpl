import os
import pandas as pd

# Path to the main directory containing camera folders
base_path = "experiments/human_nerf/zju_mocap/p387/adventure/latest/movement"

# List to store the averages of each metric for each camera file
psnr_averages = []
ssim_averages = []
mse_averages = []
mae_averages = []
lpips_averages = []

# Loop through each camera folder
for cam_id in range(1,23):  # Assuming folders are named cam_1, cam_2, ..., cam_22
    file_path = os.path.join(base_path, f"cam_{cam_id}", f"cam_{cam_id}_metrics.xlsx")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    # Read the Excel file
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # Calculate the averages for each column
    psnr_avg = df['PSNR'].mean()
    ssim_avg = df['SSIM'].mean()
    mse_avg = df['MSE'].mean()
    mae_avg = df['MAE'].mean()
    lpips_avg = df['LPIPS'].mean()
    
    # Append the averages to the respective lists
    psnr_averages.append(psnr_avg)
    ssim_averages.append(ssim_avg)
    mse_averages.append(mse_avg)
    mae_averages.append(mae_avg)
    lpips_averages.append(lpips_avg)
    
    # Append the averages as a new row at the bottom of the file
    df.loc[len(df)] = [psnr_avg, ssim_avg, mse_avg, mae_avg, lpips_avg, 'Average']
    
    # Save the modified Excel file
    df.to_excel(file_path, index=False)
    print(f"Updated file with averages: {file_path}")

# Calculate the overall average for each metric across all 23 files
overall_psnr_avg = sum(psnr_averages) / len(psnr_averages)
overall_ssim_avg = sum(ssim_averages) / len(ssim_averages)
overall_mse_avg = sum(mse_averages) / len(mse_averages)
overall_mae_avg = sum(mae_averages) / len(mae_averages)
overall_lpips_avg = sum(lpips_averages) / len(lpips_averages)

# Print the overall averages
print("Overall Averages Across All Cameras:")
print(f"PSNR: {overall_psnr_avg}")
print(f"SSIM: {overall_ssim_avg}")
print(f"MSE: {overall_mse_avg}")
print(f"MAE: {overall_mae_avg}")
print(f"LPIPS: {overall_lpips_avg}")
# Print and write the overall averages to a text file
overall_results = (
    "Overall Averages Across All Cameras:\n"
    f"PSNR: {overall_psnr_avg}\n"
    f"SSIM: {overall_ssim_avg}\n"
    f"MSE: {overall_mse_avg}\n"
    f"MAE: {overall_mae_avg}\n"
    f"LPIPS: {overall_lpips_avg}\n"
)

# Print the overall averages
print(overall_results)

# Write the overall averages to a text file
output_path = os.path.join(base_path, "evaluation.txt")
with open(output_path, "w") as f:
    f.write(overall_results)

print(f"Overall averages written to: {output_path}")
