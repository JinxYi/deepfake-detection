import io
import numpy as np
from PIL import Image

def ela(image, quality=95, scale=10):
    """
    Performs Error Level Analysis (ELA) on an image.

    Args:
        image_path (str): The path to the image file.
        scale (int): The scale factor for resizing the image. Default is 10.

    Returns:
        ela_image (np.array): The ELA image.
    """
    # Load the image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Save to JPEG in memory at given quality
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    # Compute absolute difference
    original_arr = np.array(image, dtype=np.int16)
    recompressed_arr = np.array(recompressed, dtype=np.int16)
    diff = np.abs(original_arr - recompressed_arr)

    # Scale differences
    ela_image = np.clip(diff * scale, 0, 255).astype(np.uint8)

    return Image.fromarray(ela_image)