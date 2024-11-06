import torch

# Load the checkpoint file
checkpoint = torch.load('experiments/human_nerf/zju_mocap/p387/adventure/latest.tar', map_location='cpu')
# Display available keys in the checkpoint file
print(checkpoint.keys())

# If there's a `config` or similar key
if 'config' in checkpoint:
    print("Training configuration:")
    print(checkpoint['config'])

# If specific scale or batch size information is available
if 'scale' in checkpoint:
    print("Scale used during training:", checkpoint['scale'])
if 'batch_size' in checkpoint:
    print("Batch size used during training:", checkpoint['batch_size'])
    # If there's a `config` or similar key
if 'optimizer' in checkpoint:
    print("Training configuration:")
    print(checkpoint['optimizer'])
# for key, value in checkpoint.items():
#     print(f"{key}: {value}")
