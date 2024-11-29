import re
import numpy as np
import matplotlib.pyplot as plt

def calculate_epoch_averages(epoch_losses):
    # Dictionary to store averages
    epoch_averages = {}
    
    for epoch in epoch_losses:
        epoch_averages[epoch] = {
            'total_loss': np.mean(epoch_losses[epoch]['total_loss']),
            'lpips': np.mean(epoch_losses[epoch]['lpips']),
            'mse': np.mean(epoch_losses[epoch]['mse']),
            'silhouette': np.mean(epoch_losses[epoch]['silhouette'])
        }
    
    return epoch_averages


def plot_losses(epoch_averages):
    epochs = sorted(epoch_averages.keys())
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Losses by Epoch', fontsize=16)
    
    # Plot average total loss
    total_loss = [epoch_averages[e]['total_loss'] for e in epochs]
    ax1.plot(epochs, total_loss, 'b-', label='Total Loss')
    ax1.set_title('Average Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot LPIPS and SSIM losses
    lpips = [epoch_averages[e]['lpips'] for e in epochs]
    # ssim = [epoch_averages[e]['ssim'] for e in epochs]
    ax2.plot(epochs, lpips, 'r-', label='LPIPS')
    # ax2.plot(epochs, ssim, 'g-', label='SSIM')
    ax2.set_title('LPIPS and SSIM Losses')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()
    
    # Plot MSE loss
    mse = [epoch_averages[e]['mse'] for e in epochs]
    ax3.plot(epochs, mse, 'm-', label='MSE')
    ax3.set_title('MSE Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.grid(True)
    ax3.legend()
    
    # Plot Silhouette loss
    silhouette = [epoch_averages[e]['silhouette'] for e in epochs]
    ax4.plot(epochs, silhouette, 'c-', label='Silhouette')
    ax4.set_title('Silhouette Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    return fig

# Main execution
# Define the file paths
log_file_path = 'experiments_315_FINAL/human_nerf/zju_mocap/p315/adventure/logs.txt'
output_file_path = 'experiments_315_FINAL/human_nerf/zju_mocap/p315/adventure/loss_track.txt'

# Dictionary to store losses for each epoch
epoch_losses = {}

# Regular expression to match epoch, loss components and total loss
pattern = r'Epoch:\s+(\d+).*Loss:\s+([\d.]+)\s+\[lpips:\s+([\d.]+)\s+mse:\s+([\d.]+)\s+silhouette:\s+([\d.]+)\s'

# Read and process the log file
with open(log_file_path, 'r') as file:
    log_content = file.readlines()

    # Save matched lines to output file
    with open(output_file_path, 'w') as output_file:
        for line_num, line in enumerate(log_content, start=1):
            match = re.search(pattern, line)
            if match:
                # Write to output file
                output_file.write(line)

                # Extract and store values
                epoch = int(match.group(1))
                total_loss = float(match.group(2))
                lpips = float(match.group(3))
                mse = float(match.group(4))
                silhouette = float(match.group(5))
             

                # Initialize lists for the epoch if not already present
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = {
                        'total_loss': [],
                        'lpips': [],
                        'mse': [],
                        'silhouette': []
                    }

                # Append values
                epoch_losses[epoch]['total_loss'].append(total_loss)
                epoch_losses[epoch]['lpips'].append(lpips)
                epoch_losses[epoch]['mse'].append(mse)
                epoch_losses[epoch]['silhouette'].append(silhouette)
   
            # else:
            # Print the line number and content if it does not match
            # print(f"Line {line_num} did not match the pattern: {line.strip()}")

print(len(epoch_losses[1]['total_loss']))
print(len(epoch_losses[2]['total_loss']))
# print(len(epoch_losses[3]['total_loss']))
# print(len(epoch_losses[4]['total_loss']))
# print(len(epoch_losses[5]['total_loss']))
# Calculate averages for each epoch
epoch_averages = calculate_epoch_averages(epoch_losses)

# Create and save the plots
fig = plot_losses(epoch_averages)
plt.savefig('experiments_315_FINAL/human_nerf/zju_mocap/p315/adventure/loss_analysis_per_epoch.png', dpi=300,
            bbox_inches='tight')
plt.close()

# Print average losses for each epoch
# print("\nEpoch Averages:")
# for epoch in sorted(epoch_averages.keys()):
#     print(f"\nEpoch {epoch}:")
#     for metric, value in epoch_averages[epoch].items():
#         print(f"{metric}: {value:.4f}")