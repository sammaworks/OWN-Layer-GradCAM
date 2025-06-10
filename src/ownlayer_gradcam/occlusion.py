import numpy as np
import cv2

def occlude_image(heatmap, image, occlusion_value='mean'):
    """Applies occlusion to the image based on the heatmap."""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    threshold = np.percentile(heatmap_resized, 95)
    mask = heatmap_resized > threshold

    occluded_image = image.copy()
    if occlusion_value == 'zero':
        occlusion_color = 0
    elif occlusion_value == 'mean':
        occlusion_color = np.mean(image)
    else:
        occlusion_color = occlusion_value  

    occluded_image[mask] = occlusion_color
    return occluded_image
