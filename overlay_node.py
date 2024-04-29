import cv2
import numpy as np
import tensorflow as tf

class OverlayNode:
    """
    A node that overlays image 2 on top of image 1, with black pixels in image 2 being fully transparent and white pixels being fully opaque.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "overlay"
    CATEGORY = "Image Processing"

    def overlay(self, image1, image2):
        # Convert TensorFlow Tensors to NumPy arrays if necessary
        if isinstance(image1, tf.Tensor):
            image1 = image1.numpy()
        if isinstance(image2, tf.Tensor):
            image2 = image2.numpy()

        # Ensure both images are OpenCV images
        if not isinstance(image1, np.ndarray):
            image1 = cv2.imread(image1)
        if not isinstance(image2, np.ndarray):
            image2 = cv2.imread(image2)

        # Ensure both images are in the same mode
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2BGRA)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2BGRA)

        # Create a mask from image2 where black pixels are fully transparent and white pixels are fully opaque
        mask = cv2.cvtColor(image2, cv2.COLOR_BGRA2GRAY) # Convert to grayscale
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV) # Invert colors

        # Apply the mask to image2
        image2[:, :, 3] = mask

        # Overlay image2 on top of image1
        result = cv2.addWeighted(image1, 1, image2, 1, 0)

        return result
