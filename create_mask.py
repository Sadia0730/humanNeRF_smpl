import os
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image

# Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Define the folder containing the frames
folder_path = r"/home/sadia/Downloads/CoreView_313/images"

# Define the output folder for masks
output_folder = r"/home/sadia/Downloads/CoreView_313/masks"
os.makedirs(output_folder, exist_ok=True)

# Define preprocessing transforms
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loop through all images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Load and preprocess the image
        input_image = Image.open(os.path.join(folder_path, filename))
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # Create a mask image
        mask_image = transforms.ToPILImage()(output_predictions.byte())

        # Save the mask image
        mask_path = os.path.join(output_folder, filename)
        mask_image.save(mask_path)

print("Masks created for all frames.")