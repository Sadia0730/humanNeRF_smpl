# import re
# import matplotlib.pyplot as plt

# # Define the file paths
# log_file_path = 'experiments_with_scheduler/human_nerf/zju_mocap/p387/adventure/logs.txt'
# output_file_path = 'experiments_with_scheduler/human_nerf/zju_mocap/p387/adventure/loss_track.txt'

# # Initialize lists to store iteration numbers and corresponding losses
# iterations = []
# losses = []
# ssim_losses = []
# lpips_losses = []
# silhouette_losses = []
# mse_losses = []

# # Regular expression pattern to match the line format and capture iteration and loss
# pattern = r'Epoch:.*\s\[Iter\s(\d+),.*\sLoss:\s([\d.]+)'
# pattern2 = r'Epoch:\s+(\d+).*Loss:\s+([\d.]+)\s+\[lpips:\s+([\d.]+)\s+mse:\s+([\d.]+)\s+silhouette:\s+([\d.]+)\s+ssim:\s+([\d.]+)'

# # Open the output file to save extracted lines
# with open(output_file_path, 'w') as output_file:
#     # Read the log file and extract iterations and loss values every 10,000 iterations
#     with open(log_file_path, 'r') as file:
#         for line in file:
#             match = re.search(pattern2, line)
#             if match:
#                 iter_num = int(match.group(1))  # Extract iteration number
#                 loss_val = float(match.group(2))  # Extract loss value
#                 lpips = float(match.group(3))
#                 mse = float(match.group(4))
#                 silhouette = float(match.group(5))
#                 ssim = float(match.group(6))
#                 # Save the matched line to the output file
#                 output_file.write(line)
#                 # Only store the data for every 10,000 iterations
#                 if iter_num % 1000 == 0:
#                     iterations.append(iter_num)
#                     losses.append(loss_val)
#                     ssim_losses.append(ssim)
#                     lpips_losses.append(lpips)
#                     silhouette_losses.append(silhouette)
#                     mse_losses.append(mse)


# # # Plotting the training loss over iterations
# # plt.figure(figsize=(20, 16))
# # plt.plot(iterations, losses, label='Training Loss', color='blue')
# # plt.xlabel('Iteration')
# # plt.ylabel('Loss')
# # plt.title('Training Loss over Iterations (Every 5,000 Iterations)')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig('experiments_with_scheduler/human_nerf/zju_mocap/p387/adventure/training_loss_graph.png')  # Save the plot
# # plt.show()
# # Plotting each loss on a subplot
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# fig.suptitle('Training Loss Over Iterations', fontsize=16)

# # Subplot 0,0: Total Loss
# axes[0, 0].plot(iterations, losses, label='Total Loss', color='blue')
# axes[0, 0].set_title('Total Loss')
# axes[0, 0].set_xlabel('Iteration')
# axes[0, 0].set_ylabel('Loss')
# axes[0, 0].grid(True)

# # Subplot 0,1: LPIPS and SSIM Loss
# axes[0, 1].plot(iterations, lpips_losses, label='LPIPS Loss', color='orange')
# axes[0, 1].plot(iterations, ssim_losses, label='SSIM Loss', color='green')
# axes[0, 1].set_title('LPIPS and SSIM Loss')
# axes[0, 1].set_xlabel('Iteration')
# axes[0, 1].set_ylabel('Loss')
# axes[0, 1].legend()
# axes[0, 1].grid(True)

# # Subplot 1,0: MSE Loss
# axes[1, 0].plot(iterations, mse_losses, label='MSE Loss', color='purple')
# axes[1, 0].set_title('MSE Loss')
# axes[1, 0].set_xlabel('Iteration')
# axes[1, 0].set_ylabel('Loss')
# axes[1, 0].grid(True)

# # Subplot 1,1: Silhouette Loss
# axes[1, 1].plot(iterations, silhouette_losses, label='Silhouette Loss', color='red')
# axes[1, 1].set_title('Silhouette Loss')
# axes[1, 1].set_xlabel('Iteration')
# axes[1, 1].set_ylabel('Loss')
# axes[1, 1].grid(True)

# # Save and show the plot
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title space
# plt.savefig('experiments_with_scheduler/human_nerf/zju_mocap/p387/adventure/loss_plots.png')
# plt.show()
import re
import matplotlib.pyplot as plt

# Define file paths
log_file_path = 'experiments_with_scheduler/human_nerf/zju_mocap/p387/adventure/logs.txt'
output_file_path = 'experiments_with_scheduler/human_nerf/zju_mocap/p387/adventure/loss_track.txt'

# Initialize lists to store values
iterations = []
losses = []
lpips_losses = []
mse_losses = []
silhouette_losses = []

# Updated regex pattern
pattern2 = r'Epoch:\s+\d+\s+\[Iter\s+(\d+),.*Loss:\s+([\d.]+)\s+\[lpips:\s+([\d.]+)\s+mse:\s+([\d.]+)\s+silhouette:\s+([\d.]+)'

# Extract data from log file
with open(output_file_path, 'w') as output_file:
    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(pattern2, line)
            if match:
                iter_num = int(match.group(1))
                loss_val = float(match.group(2))
                lpips = float(match.group(3))
                mse = float(match.group(4))
                silhouette = float(match.group(5))
                
                # Save to output file and append to lists every 1000 iterations
                output_file.write(line)
                if iter_num % 1000 == 0:
                    iterations.append(iter_num)
                    losses.append(loss_val)
                    lpips_losses.append(lpips)
                    mse_losses.append(mse)
                    silhouette_losses.append(silhouette)

# Plot data in subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Loss Over Iterations', fontsize=16)

# Subplot 0,0: Total Loss
axes[0, 0].plot(iterations, losses, label='Total Loss', color='blue')
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True)

# Subplot 0,1: LPIPS Loss
axes[0, 1].plot(iterations, lpips_losses, label='LPIPS Loss', color='orange')
axes[0, 1].set_title('LPIPS Loss')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True)

# Subplot 1,0: MSE Loss
axes[1, 0].plot(iterations, mse_losses, label='MSE Loss', color='purple')
axes[1, 0].set_title('MSE Loss')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True)

# Subplot 1,1: Silhouette Loss
axes[1, 1].plot(iterations, silhouette_losses, label='Silhouette Loss', color='red')
axes[1, 1].set_title('Silhouette Loss')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].grid(True)

# Save and display plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('experiments_with_scheduler/human_nerf/zju_mocap/p387/adventure/loss_plots.png')
plt.show()

