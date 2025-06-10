import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from .occlusion import occlude_image
from .image_utils import read_image
from .layer_importance import evaluate_layer_importance
from .visualization import display_results

class OWNLayerGradCAM:
    def __init__(self, model, n_layers=8, erosion_iterations=1):
        self.model = model
        self.n_layers = n_layers
        self.erosion_iterations = erosion_iterations
        self.total_conv_layers = len([layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)])
        self.n_layers = min(self.n_layers, self.total_conv_layers)

    def grad_cam(self, image, true_label, layer_conv_name):
        """Generates Grad-CAM heatmap for a specific convolutional layer."""
        model_grad = Model(inputs=self.model.input,
                           outputs=[self.model.get_layer(layer_conv_name).output, self.model.output])

        with tf.GradientTape() as tape:
            conv_output, predictions = model_grad(image)
            tape.watch(conv_output)
            loss = tf.losses.categorical_crossentropy(true_label, predictions)

        grad = tape.gradient(loss, conv_output)
        grad = K.mean(tf.abs(grad), axis=(0, 1, 2))
        conv_output = np.squeeze(conv_output.numpy())

        # Apply gradient weights to the convolutional output
        for i in range(conv_output.shape[-1]):
            conv_output[:, :, i] *= grad[i]

        # Average across all channels to get a single heatmap
        heatmap = tf.reduce_mean(conv_output, axis=-1)
        return np.squeeze(heatmap), np.squeeze(image)

    def get_final_conv_layers(self):
        """Returns the last n convolutional layers."""
        conv_layers = [layer for layer in self.model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        return conv_layers[-self.n_layers:]

    def generate_weighted_heatmap(self, image_tb, label_tb):
        """Generates a weighted heatmap by combining heatmaps from different layers."""
        final_conv_layers = self.get_final_conv_layers()
        heatmaps = [self.grad_cam(image_tb, label_tb, layer.name)[0] for layer in final_conv_layers]
        
        weights = evaluate_layer_importance(self, image_tb, heatmaps, label_tb)
        weights = np.abs(weights)
        weights = tf.reshape(weights, (-1, 1, 1))

        # Stack and weight the heatmaps
        stacked_heatmaps = tf.stack(heatmaps, axis=0)
        weighted_sum = tf.reduce_sum(stacked_heatmaps * weights, axis=0)
        weighted_average = weighted_sum / tf.reduce_sum(weights)

        # Normalize the final heatmap
        average_heatmap = np.maximum(weighted_average, 0)
        average_heatmap = average_heatmap / tf.reduce_max(average_heatmap)
        average_heatmap = cv2.resize(average_heatmap.numpy(), (224, 224))

        return average_heatmap

    def generate_and_display_heatmap(self, images_path, true_label, kernel_size=3):
        """Generates and displays the Grad-CAM heatmap for an image."""
        image_tb = read_image(images_path)
        average_heatmap = self.generate_weighted_heatmap(image_tb, true_label)
        refined_heatmap = occlude_image(average_heatmap, image_tb[0])
        display_results(image_tb[0], refined_heatmap)
