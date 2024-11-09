# import os
# import pandas as pd
#
# # Path to the main directory containing camera folders
# base_path = "experiments_with_nan_backup/human_nerf/zju_mocap/p387/adventure/latest/movement_0.2"
#
# # Threshold values for comparison (from HumanNeRF)
# psnr_threshold = 28.18
# ssim_threshold = 0.9632
# lpips_threshold = 0.03558
#
# # Lists to store camera names for each metric if they exceed HumanNeRF's values
# psnr_better = []
# ssim_better = []
# lpips_better = []
#
# # Loop through each camera folder
# for cam_id in range(1, 23):  # Assuming folders are named cam_1, cam_2, ..., cam_22
#     file_path = os.path.join(base_path, f"cam_{cam_id}", f"cam_{cam_id}_metrics.xlsx")
#
#     # Check if the file exists
#     if not os.path.exists(file_path):
#         print(f"File not found: {file_path}")
#         continue
#
#     # Read the Excel file
#     df = pd.read_excel(file_path)
#
#     # Find the row with "Average" in the 6th column
#     avg_row = df[df.iloc[:, 5] == "Average"]
#
#     # Check if the average row exists
#     if not avg_row.empty:
#         psnr_avg = avg_row['PSNR'].values[0]
#         ssim_avg = avg_row['SSIM'].values[0]
#         lpips_avg = avg_row['LPIPS'].values[0]
#
#         # Compare and store folder name if the metric is better than HumanNeRF
#         if psnr_avg > psnr_threshold:
#             psnr_better.append(f"cam_{cam_id}")
#         if ssim_avg > ssim_threshold:
#             ssim_better.append(f"cam_{cam_id}")
#         if lpips_avg < lpips_threshold:  # LPIPS should be lower
#             lpips_better.append(f"cam_{cam_id}")
#     else:
#         print(f"No 'Average' row found in file: {file_path}")
#
# # Write the results to a text file
# output_file = "cam_better_than_human_nerf.txt"
# with open(output_file, "w") as f:
#     f.write("PSNR: " + ", ".join(psnr_better) + "\n")
#     f.write("SSIM: " + ", ".join(ssim_better) + "\n")
#     f.write("LPIPS: " + ", ".join(lpips_better) + "\n")
#
# print(f"Results saved to {output_file}")

import os
import pandas as pd

# Path to the main directory containing camera folders
base_path = "experiments_with_nan_backup/human_nerf/zju_mocap/p387/adventure/latest/movement_0.2"

# Threshold values for comparison (from HumanNeRF)
psnr_threshold = 28.18
ssim_threshold = 0.9632
lpips_threshold = 0.03558

# Dictionaries to store the results for each metric
psnr_better = {}
ssim_better = {}
lpips_better = {}

# Loop through each camera folder
for cam_id in range(1, 23):  # Assuming folders are named cam_1, cam_2, ..., cam_22
    file_path = os.path.join(base_path, f"cam_{cam_id}", f"cam_{cam_id}_metrics.xlsx")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Read the Excel file
    df = pd.read_excel(file_path)

    # Filter rows up to the last image row (541) and ignore the 'Average' row
    data_rows = df.iloc[:541]  # Assuming the "Average" row is the last row (542)

    # Check PSNR, SSIM, and LPIPS for each image row
    psnr_indices = []
    ssim_indices = []
    lpips_indices = []

    for _, row in data_rows.iterrows():
        image_index = row.iloc[5]  # Get the image index from the 6th column

        # Check PSNR
        if row['PSNR'] > psnr_threshold:
            psnr_indices.append(image_index)

        # Check SSIM
        if row['SSIM'] > ssim_threshold:
            ssim_indices.append(image_index)

        # Check LPIPS (lower is better)
        if row['LPIPS'] < lpips_threshold:
            lpips_indices.append(image_index)

    # Store results if any images in this camera folder exceed the thresholds
    if psnr_indices:
        psnr_better[f"cam_{cam_id}"] = psnr_indices
    if ssim_indices:
        ssim_better[f"cam_{cam_id}"] = ssim_indices
    if lpips_indices:
        lpips_better[f"cam_{cam_id}"] = lpips_indices

# Write the results to a text file
output_file = "better_than_human_nerf_per_image.txt"
with open(output_file, "w") as f:
    # PSNR Results
    f.write("PSNR: " + ", ".join([f"{cam} ({len(indices)})" for cam, indices in psnr_better.items()]) + "\n")
    for cam, indices in psnr_better.items():
        f.write(f"  {cam}: {', '.join(map(str, indices))}\n")

    # SSIM Results
    f.write("\nSSIM: " + ", ".join([f"{cam} ({len(indices)})" for cam, indices in ssim_better.items()]) + "\n")
    for cam, indices in ssim_better.items():
        f.write(f"  {cam}: {', '.join(map(str, indices))}\n")

    # LPIPS Results
    f.write("\nLPIPS: " + ", ".join([f"{cam} ({len(indices)})" for cam, indices in lpips_better.items()]) + "\n")
    for cam, indices in lpips_better.items():
        f.write(f"  {cam}: {', '.join(map(str, indices))}\n")

print(f"Results saved to {output_file}")

