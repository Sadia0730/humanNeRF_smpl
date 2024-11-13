import os
import pandas as pd

# Paths to the main directories containing camera folders for both versions
base_path_1 = "experiments_with_nan_backup_without non_rigid/human_nerf/zju_mocap/p387/adventure/latest/movement_0.5"
base_path_2 = "experiments_with_nan_backup/human_nerf_old/zju_mocap/p387/adventure/latest/movement_0.5"

# Dictionaries to store the results for each metric
psnr_better = {}
ssim_better = {}
lpips_better = {}
best_metrics = {}

# Loop through each camera folder
for cam_id in range(1, 23):  # Assuming folders are named cam_1, cam_2, ..., cam_22
    file_path_1 = os.path.join(base_path_1, f"cam_{cam_id}", f"cam_{cam_id}_metrics.xlsx")
    file_path_2 = os.path.join(base_path_2, f"cam_{cam_id}", f"cam_{cam_id}_metrics.xlsx")

    # Check if both files exist
    if not os.path.exists(file_path_1) or not os.path.exists(file_path_2):
        print(f"File not found: {file_path_1} or {file_path_2}")
        continue

    # Read the Excel files
    df1 = pd.read_excel(file_path_1)
    df2 = pd.read_excel(file_path_2)

    # Filter rows up to the last image row (541) and ignore the 'Average' row
    data_rows_1 = df1.iloc[:541]
    data_rows_2 = df2.iloc[:541]

    # Check PSNR, SSIM, and LPIPS for each image row
    psnr_indices = []
    ssim_indices = []
    lpips_indices = []

    best_psnr_diff, best_ssim_diff, best_lpips_diff = -float('inf'), -float('inf'), float('inf')
    best_psnr_index, best_ssim_index, best_lpips_index = None, None, None

    for idx, (row1, row2) in enumerate(zip(data_rows_1.iterrows(), data_rows_2.iterrows())):
        image_index = row1[1].iloc[5]  # Get the image index from the 6th column

        # Check PSNR
        psnr_diff = row1[1]['PSNR'] - row2[1]['PSNR']
        if psnr_diff > 0:
            psnr_indices.append(image_index)
            if psnr_diff > best_psnr_diff:
                best_psnr_diff = psnr_diff
                best_psnr_index = image_index

        # Check SSIM
        ssim_diff = row1[1]['SSIM'] - row2[1]['SSIM']
        if ssim_diff > 0:
            ssim_indices.append(image_index)
            if ssim_diff > best_ssim_diff:
                best_ssim_diff = ssim_diff
                best_ssim_index = image_index

        # Check LPIPS (lower is better)
        lpips_diff = row2[1]['LPIPS'] - row1[1]['LPIPS']
        if lpips_diff > 0:
            lpips_indices.append(image_index)
            if lpips_diff < best_lpips_diff:
                best_lpips_diff = lpips_diff
                best_lpips_index = image_index

    # Store results if any images in this camera folder exceed the metrics in the second file
    if psnr_indices:
        psnr_better[f"cam_{cam_id}"] = psnr_indices
        best_metrics[f"cam_{cam_id}_best_psnr"] = (best_psnr_index, best_psnr_diff)
    if ssim_indices:
        ssim_better[f"cam_{cam_id}"] = ssim_indices
        best_metrics[f"cam_{cam_id}_best_ssim"] = (best_ssim_index, best_ssim_diff)
    if lpips_indices:
        lpips_better[f"cam_{cam_id}"] = lpips_indices
        best_metrics[f"cam_{cam_id}_best_lpips"] = (best_lpips_index, best_lpips_diff)

# Write the results to a text file
output_file = os.path.join(base_path_1, "comparison_better_results_per_image.txt")
with open(output_file, "w") as f:
    # PSNR Results
    f.write("PSNR: " + ", ".join([f"{cam} ({len(indices)})" for cam, indices in psnr_better.items()]) + "\n")
    for cam, indices in psnr_better.items():
        f.write(f"  {cam}: {', '.join(map(str, indices))}\n")
        f.write(
            f"  Best PSNR Image: {best_metrics[f'{cam}_best_psnr'][0]} with Difference: {best_metrics[f'{cam}_best_psnr'][1]}\n")

    # SSIM Results
    f.write("\nSSIM: " + ", ".join([f"{cam} ({len(indices)})" for cam, indices in ssim_better.items()]) + "\n")
    for cam, indices in ssim_better.items():
        f.write(f"  {cam}: {', '.join(map(str, indices))}\n")
        f.write(
            f"  Best SSIM Image: {best_metrics[f'{cam}_best_ssim'][0]} with Difference: {best_metrics[f'{cam}_best_ssim'][1]}\n")

    # LPIPS Results
    f.write("\nLPIPS: " + ", ".join([f"{cam} ({len(indices)})" for cam, indices in lpips_better.items()]) + "\n")
    for cam, indices in lpips_better.items():
        f.write(f"  {cam}: {', '.join(map(str, indices))}\n")
        f.write(
            f"  Best LPIPS Image: {best_metrics[f'{cam}_best_lpips'][0]} with Difference: {best_metrics[f'{cam}_best_lpips'][1]}\n")

    # Intersection of PSNR, SSIM, and LPIPS better images
    f.write("\nIntersection of Better Images in PSNR, SSIM, and LPIPS:\n")
    for cam in psnr_better.keys():
        psnr_set = set(psnr_better.get(cam, []))
        ssim_set = set(ssim_better.get(cam, []))
        lpips_set = set(lpips_better.get(cam, []))

        intersection = psnr_set & ssim_set & lpips_set
        if intersection:
            f.write(f"  {cam}: {', '.join(map(str, intersection))}\n")

print(f"Results saved to {output_file}")
