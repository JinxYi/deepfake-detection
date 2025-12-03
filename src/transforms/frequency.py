import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
import pywt   # wavelet
import torch.nn.functional as F

RESNET_INPUT_MEAN = [0.485, 0.456, 0.406]
RESNET_INPUT_SD = [0.229, 0.224, 0.225]
RESNET_IMAGE_SIZE = (224, 224)

epsilon = 1e-8

def fft_magnitude(img: Image.Image):
    img = np.array(img.convert("L"), dtype=np.float32)  # grayscale
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    mag_range = mag.max() - mag.min()

    if mag_range < epsilon:
        print("Warning: magnitude has close to zero range.")
        mag = np.zeros_like(mag)
    else:
        mag = (mag - mag.min()) / mag_range

    mag = mag.astype(np.float32)
    return torch.from_numpy(mag).unsqueeze(0)  # shape (1, H, W)

def fft_phase(img: Image.Image):
    img = np.array(img.convert("L"), dtype=np.float32)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    phase = np.angle(fshift)
    phase = (phase + np.pi) / (2 * np.pi)

    phase = phase.astype(np.float32)  # (1,H,W)
    return torch.from_numpy(phase).unsqueeze(0)

def fft_amp_phase(img: Image.Image):
    img = np.array(img.convert("L"), dtype=np.float32)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    amp = np.log1p(np.abs(fshift))

    amp_range = amp.max() - amp.min()
    if amp_range < epsilon:
        print("Warning: amp part has close to zero range.")
        amp = np.zeros_like(amp)
    else:
        amp = (amp - amp.min()) / amp_range

    phase = np.angle(fshift)
    phase = (phase + np.pi) / (2 * np.pi)

    arr = np.stack([amp, phase], axis=0).astype(np.float32)  # (2,H,W)
    return torch.from_numpy(arr)


def dct_coeff(img: Image.Image):
    img = np.array(img.convert("L"), dtype=np.float32)
    dct = cv2.dct(img)
    dct = np.log1p(np.abs(dct))
    dct_range = dct.max() - dct.min()

    if dct_range < epsilon:
        print("Warning: DCT has close to zero range.")
        dct = np.zeros_like(dct)
    else:
        dct = (dct - dct.min()) / dct_range
    dct = dct.astype(np.float32)
    return torch.from_numpy(dct).unsqueeze(0)  # (1, H, W)


def wavelet_subbands(img: Image.Image, wavelet='haar', upsample=True, size=RESNET_IMAGE_SIZE) -> torch.Tensor:
    img = np.array(img.convert("L"), dtype=np.float32)
    coeffs2 = pywt.dwt2(img, wavelet)
    cA, (cH, cV, cD) = coeffs2

    def norm(x):
        x_range = x.max() - x.min()
        if x_range < epsilon:
            print("Warning: wavelet subband has close to zero range.")
            x = np.zeros_like(x)
        else:
            x = (x - x.min()) / x_range
        return x

    arr = np.stack(
        [norm(cA), norm(cH), norm(cV), norm(cD)], axis=0
    ).astype(np.float32)  # (4, H/2, W/2)
    tensor = torch.from_numpy(arr)

    if upsample:
        tensor = F.interpolate(tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)
    return tensor  # (4, H, W)

def dwt_rgb_concat(img: Image.Image, image_size=224):
    rgb_t = transforms.ToTensor()(img)           # (3, H, W)
    wave_t = wavelet_subbands(img, upsample=True, size=(image_size, image_size))  # (4, H, W)
    return torch.cat([rgb_t, wave_t], dim=0)     # (7, H, W)

def high_freq_residual(img: Image.Image, kernel_size=5) -> torch.Tensor:
    img_gray = np.array(img.convert("L"), dtype=np.float32)
    blur = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
    res = img_gray - blur
    res_range = res.max() - res.min()
    if res_range < epsilon:
        print("Warning: residual has close to zero range.")
        res = np.zeros_like(res)
    else:
        res = (res - res.min()) / res_range
    res = res.astype(np.float32)
    return torch.from_numpy(res).unsqueeze(0)  # (1, H, W)



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
        freq_transform = transforms.ToTensor()
    elif mode == 'fft_mag':
        norm_mean, norm_std = [0.5], [0.5]
        freq_transform = transforms.Lambda(fft_magnitude)
    
    elif mode == 'fft_phase':
        norm_mean, norm_std = [0.5], [0.5]
        freq_transform = transforms.Lambda(fft_phase)

    elif mode == 'fft_mag_phase':
        norm_mean, norm_std = [0.5, 0.5], [0.5, 0.5]
        freq_transform = transforms.Lambda(fft_amp_phase)

    elif mode == 'dct':
        norm_mean, norm_std = [0.5], [0.5]
        freq_transform = transforms.Lambda(dct_coeff)

    elif mode == 'wavelet':
        norm_mean, norm_std = [0.5]*4, [0.5]*4
        freq_transform = transforms.Lambda(wavelet_subbands)

    elif mode == 'dwt_rgb':
        norm_mean = RESNET_INPUT_MEAN + [0.5]*4
        norm_std = RESNET_INPUT_SD + [0.5]*4
        freq_transform = transforms.Lambda(lambda img: dwt_rgb_concat(img, image_size=image_size))
    elif mode == 'residual':
        norm_mean, norm_std = [0.5], [0.5]
        freq_transform = transforms.Lambda(high_freq_residual)

    else:
        raise ValueError(f"Unknown transform mode: {mode}")

    data_transforms = {
        'train': transforms.Compose(common_train + [
            freq_transform,
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'val': transforms.Compose(common_eval + [
            freq_transform,
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'test': transforms.Compose(common_eval + [
            freq_transform,
            transforms.Normalize(norm_mean, norm_std)
        ]),
    }

    return data_transforms
