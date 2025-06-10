import numpy as np
import tensorflow as tf

def evaluate_layer_importance(gradcam, image, heatmaps, true_label):
    """Evaluates the importance of each layer based on the heatmap."""
    original_pred = gradcam.model.predict(image)[0]
    target_class = np.argmax(true_label)
    original_prob = original_pred[target_class]

    importance_scores = []
    for heatmap in heatmaps:
        occluded_image = gradcam.occlude_image(image[0], heatmap, occlusion_value='mean')
        occluded_image = np.expand_dims(occluded_image, axis=0)
        occluded_pred = gradcam.model.predict(occluded_image)[0]
        occluded_prob = occluded_pred[target_class]
        importance_score = original_prob - occluded_prob
        importance_scores.append(importance_score)

    importance_scores = np.array(importance_scores)
    weights = importance_scores / np.sum(importance_scores)

    return weights