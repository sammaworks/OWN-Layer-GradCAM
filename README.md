# OWN-Layer Grad-CAM Project

This project implements OWN-Layer Grad-CAM (Occlusion-Weighted N-Layer Gradient-weighted Class Activation Mapping), a technique for visualizing the regions of an image that are important for a model's predictions. It provides a way to understand and interpret deep learning models, particularly convolutional neural networks (CNNs).

## Project Structure

The project is organized into several directories and files:

- **data/**: Contains sample images and data for testing and demonstration.
  - **sample_images/**: Directory for storing sample images.
  - **`__init__.py`**: Makes the data directory a Python package.

- **notebooks/**: Jupyter Notebooks for quick experimentation.
  - **demo_ownlayer_gradcam.ipynb**: Example notebook for running OWN-Layer Grad-CAM with a model.

- **src/**: Source code for the project.
  - **`__init__.py`**: Makes the src directory a Python package.
  - **ownlayer_gradcam/**: Core OWN-Layer Grad-CAM functionality.
    - **`__init__.py`**: Initializes the gradcam package.
    - **ownlayer_gradcam.py**: Main code for OWN-Layer Grad-CAM functionality.
    - **occlusion.py**: Handles occlusion logic and image masking.
    - **image_utils.py**: Utilities for image loading and preprocessing.
    - **layer_importance.py**: Evaluates layer importance.
    - **visualization.py**: Visualizes OWN-Layer Grad-CAM results.
  - **utils/**: General utility functions.
    - **`__init__.py`**: Initializes the utils package.
    - **model_utils.py**: Helper functions related to model loading and training.
    - **data_utils.py**: Functions for loading datasets.
    - **logging.py**: Logging setup and utilities.

- **tests/**: Unit tests for the project.
  - **`__init__.py`**: Initializes the tests package.
  - **test_ownlayer_gradcam.py**: Unit tests for OWN-Layer Grad-CAM class and methods.
  - **test_occlusion.py**: Unit tests for image occlusion logic.

- **examples/**: Example scripts and demos for using the OWN-Layer Grad-CAM code.
  - **ownlayer_gradcam_demo.py**: Demonstrates how to use the OWN-Layer Grad-CAM class on an image.
  - **ownlayer_gradcam_model_example.py**: Example of applying OWN-Layer Grad-CAM to a trained model.

- **requirements.txt**: Lists dependencies required for the project.

- **README.md**: Documentation and usage guide for the project.

- **setup.py**: Setup script for the project.

- **LICENSE**: Project license (e.g., MIT, GPL).

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset and place sample images in the `data/sample_images/` directory.
2. Use the provided Jupyter Notebook `notebooks/demo_ownlayer_gradcam.ipynb` for quick experimentation with OWN-Layer Grad-CAM.
3. Refer to the example scripts in the `examples/` directory for practical implementations.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.