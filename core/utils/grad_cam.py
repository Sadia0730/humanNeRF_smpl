import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax().item()

        self.model.zero_grad()
        output[:, target_class].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3], keepdim=True)
        cam = torch.sum(self.activations * pooled_gradients, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)  # Apply ReLU
        cam = cam / cam.max()

        return cam


