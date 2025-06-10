import unittest
from src.gradcam.gradcam import GradCAM

class TestGradCAM(unittest.TestCase):
    def test_gradcam_initialization(self):
        """Test if GradCAM initializes correctly."""
        gradcam = GradCAM(model=None, layer_name="conv_layer")
        self.assertIsNotNone(gradcam)