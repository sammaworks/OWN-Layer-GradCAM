from src.gradcam.gradcam import OWNLayerGradCAM
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import numpy as np

# Load a pre-trained model (VGG16 for example)
model = VGG16(weights='imagenet')

# Initialize Grad-CAM
ownlayer_grad_cam = NLayerGradCAM(model)

# Example image and label (to be replaced with actual image and label)
img_path = 'path_to_image.jpg'  # Replace with an actual image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
true_label = np.zeros((1, 1000))  # Example one-hot encoded label
true_label[0, 386] = 1  # Simulate the label of the correct class

# Generate and display the heatmap
ownlayer_grad_cam.generate_and_display_heatmap(img_path, true_label)
