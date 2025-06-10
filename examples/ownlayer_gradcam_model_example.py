# Import necessary libraries
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.gradcam.gradcam import OWNLayerGradCAM
from src.utils.model_utils import load_model

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to display the image and the Grad-CAM result
def display_results(original_image, heatmap, cam_image):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Grad-CAM Result")
    plt.imshow(cam_image)
    plt.axis('off')

    plt.show()

# Main function to run the Grad-CAM example
def main(image_path):
    # Load the pre-trained model
    model = load_model(models.resnet50(pretrained=True))
    model.eval()

    # Load and preprocess the image
    input_image = load_and_preprocess_image(image_path)
    
    # Create Grad-CAM object
    grad_cam = GradCAM(model)

    # Generate the heatmap
    heatmap = grad_cam.generate_heatmap(input_image)

    # Convert the heatmap to a format suitable for visualization
    original_image = Image.open(image_path).convert("RGB")
    heatmap = np.uint8(255 * heatmap.squeeze())
    heatmap = Image.fromarray(heatmap).resize(original_image.size)
    cam_image = Image.blend(original_image, heatmap.convert("RGB"), alpha=0.5)

    # Display the results
    display_results(original_image, heatmap, cam_image)

# Entry point of the script
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    main(image_path)