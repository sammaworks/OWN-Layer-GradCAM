import unittest
from src.gradcam.occlusion import apply_occlusion

class TestOcclusion(unittest.TestCase):
    def test_apply_occlusion(self):
        """Test if occlusion is applied correctly."""
        result = apply_occlusion(image=None, mask=None)
        self.assertIsNotNone(result)