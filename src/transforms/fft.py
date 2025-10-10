from PIL import Image
import numpy as np

def fft(image):
    """
    Applies Fast Fourier Transform (FFT) to an image and returns the magnitude spectrum.

    Args:
        image (PIL.Image or np.array): Input image.
    """
    # Load the image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_np = np.array(image)

    # Compute 2D FFT
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)  # Shift zero freq to center

    # Get magnitude spectrum (frequency information)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)  # Add epsilon to avoid log(0)
    phase = np.angle(fshift)

    magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    phase_norm = (phase - phase.min()) / (phase.max() - phase.min() + 1e-8)

    # Optionally, reduce to single channel (e.g., mean across RGB)
    # magnitude_norm = magnitude_norm.mean(axis=2, keepdims=True)
    # phase_norm = phase_norm.mean(axis=2, keepdims=True)

    # Stack with original image
    img_norm = image / 255.0  # normalize image to [0, 1]
    combined = np.concatenate([img_norm, magnitude_norm, phase_norm], axis=2)  # shape: (H, W, 9) if RGB

    # If your model expects (C, H, W):
    combined = np.transpose(combined, (2, 0, 1))
    return combined

