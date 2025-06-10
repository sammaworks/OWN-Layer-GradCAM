import cv2
import numpy as np

def read_image(images_path):
    """Reads and preprocesses the image."""
    img = cv2.imread(str(images_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize the image to [0, 1]
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return np.array(img)