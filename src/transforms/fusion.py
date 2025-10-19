import torch
from torchvision import transforms
from .frequency import wavelet_subbands, RESNET_INPUT_MEAN, RESNET_INPUT_SD

def get_fused_transform(image_size=224):
    rgb_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    wavelet_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.Lambda(lambda img: wavelet_subbands(img, 'haar', upsample=True, size=(image_size, image_size))),
    ])

    def combined_transform(img):
        rgb = rgb_transform(img)
        wave = wavelet_transform(img)
        fused = torch.cat([rgb, wave], dim=0)  # (7, H, W)
        return fused

    # --- Augmentations ---
    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(combined_transform),
        transforms.Normalize(mean=RESNET_INPUT_MEAN + [0.5]*4,
                             std=RESNET_INPUT_SD + [0.5]*4)
    ])

    eval_aug = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.Lambda(combined_transform),
        transforms.Normalize(mean=RESNET_INPUT_MEAN + [0.5]*4,
                             std=RESNET_INPUT_SD + [0.5]*4)
    ])

    # --- Final dictionary ---
    data_transforms = {
        'train': train_aug,
        'val': eval_aug,
        'test': eval_aug,
    }

    return data_transforms