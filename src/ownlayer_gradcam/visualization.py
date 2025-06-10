import matplotlib.pyplot as plt

def display_results(original_image, heatmap):
    """Displays the original image and the heatmap overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[1].imshow(original_image)
    axes[1].imshow(heatmap, alpha=0.5, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    plt.show()