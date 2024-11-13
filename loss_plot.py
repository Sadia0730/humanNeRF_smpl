import re
import matplotlib.pyplot as plt

# Define the file paths
log_file_path = 'experiments_with_scheduler/human_nerf/zju_mocap/p387/adventure/logs.txt'
output_file_path = 'experiments_with_scheduler/human_nerf/zju_mocap/p387/adventure/loss_track.txt'

# Initialize lists to store iteration numbers and corresponding losses
iterations = []
losses = []

# Regular expression pattern to match the line format and capture iteration and loss
pattern = r'Epoch:.*\s\[Iter\s(\d+),.*\sLoss:\s([\d.]+)'

# Open the output file to save extracted lines
with open(output_file_path, 'w') as output_file:
    # Read the log file and extract iterations and loss values every 10,000 iterations
    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                iter_num = int(match.group(1))  # Extract iteration number
                loss_val = float(match.group(2))  # Extract loss value
                # Save the matched line to the output file
                output_file.write(line)
                # Only store the data for every 10,000 iterations
                if iter_num % 1000 == 0:
                    iterations.append(iter_num)
                    losses.append(loss_val)


# Plotting the training loss over iterations
plt.figure(figsize=(20, 16))
plt.plot(iterations, losses, label='Training Loss', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss over Iterations (Every 5,000 Iterations)')
plt.legend()
plt.grid(True)
plt.savefig('experiments_with_scheduler/human_nerf/zju_mocap/p387/adventure/training_loss_graph.png')  # Save the plot
plt.show()
