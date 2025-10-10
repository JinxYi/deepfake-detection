import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
import pywt   # wavelet

def fft_magnitude(img: Image.Image):
    img = np.array(img.convert("L"))  # grayscale
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    mag = (mag - mag.min()) / (mag.max() - mag.min())
    return Image.fromarray((mag * 255).astype(np.uint8))


def fft_real_imag(img: Image.Image):
    img = np.array(img.convert("L"))
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    real = np.real(fshift)
    imag = np.imag(fshift)
    real = (real - real.min()) / (real.max() - real.min())
    imag = (imag - imag.min()) / (imag.max() - imag.min())
    arr = np.stack([real, imag], axis=0).astype(np.float32)  # (2,H,W)
    return torch.from_numpy(arr)


def dct_coeff(img: Image.Image):
    img = np.array(img.convert("L"), dtype=np.float32)
    dct = cv2.dct(img)
    dct = np.log1p(np.abs(dct))
    dct = (dct - dct.min()) / (dct.max() - dct.min())
    return Image.fromarray((dct * 255).astype(np.uint8))


def wavelet_subbands(img: Image.Image, wavelet='haar'):
    img = np.array(img.convert("L"), dtype=np.float32)
    coeffs2 = pywt.dwt2(img, wavelet)
    cA, (cH, cV, cD) = coeffs2
    def norm(x):
        x = (x - x.min()) / (x.max() - x.min())
        return x
    arr = np.stack([norm(cA), norm(cH), norm(cV), norm(cD)], axis=0).astype(np.float32)
    return torch.from_numpy(arr)   # shape (4,H/2,W/2)


def high_freq_residual(img: Image.Image, kernel_size=5):
    img_gray = np.array(img.convert("L"), dtype=np.float32)
    blur = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
    res = img_gray - blur
    res = (res - res.min()) / (res.max() - res.min())
    return Image.fromarray((res * 255).astype(np.uint8))


def get_transforms(mode: str, image_size: int = 224):
    """
    mode: one of ['rgb', 'fft_mag', 'fft_ri', 'dct', 'wavelet', 'residual']
    """
    common_train = [
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
    ]
    common_eval = [
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
    ]

    if mode == 'rgb':
        norm_mean, norm_std = RESNET_INPUT_MEAN, RESNET_INPUT_SD
        freq_transform = transforms.Lambda(lambda x: x)

    elif mode == 'fft_mag':
        norm_mean, norm_std = [0.5], [0.5]
        freq_transform = transforms.Lambda(fft_magnitude)

    elif mode == 'fft_ri':
        norm_mean, norm_std = [0.5, 0.5], [0.5, 0.5]
        freq_transform = transforms.Lambda(fft_real_imag)

    elif mode == 'dct':
        norm_mean, norm_std = [0.5], [0.5]
        freq_transform = transforms.Lambda(dct_coeff)

    elif mode == 'wavelet':
        norm_mean, norm_std = [0.5]*4, [0.5]*4
        freq_transform = transforms.Lambda(wavelet_subbands)

    elif mode == 'residual':
        norm_mean, norm_std = [0.5], [0.5]
        freq_transform = transforms.Lambda(high_freq_residual)

    else:
        raise ValueError(f"Unknown transform mode: {mode}")

    data_transforms = {
        'train': transforms.Compose(common_train + [
            freq_transform,
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'val': transforms.Compose(common_eval + [
            freq_transform,
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'test': transforms.Compose(common_eval + [
            freq_transform,
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
    }

    return data_transforms
