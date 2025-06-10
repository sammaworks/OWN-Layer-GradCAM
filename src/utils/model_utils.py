import tensorflow as tf

def load_model(model_path):
    """Loads a pre-trained model from the given path."""
    return tf.keras.models.load_model(model_path)

def save_model(model, save_path):
    """Saves the model to the specified path."""
    model.save(save_path)