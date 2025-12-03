MODEL_REGISTRRY = {
    "dct_resnet18": {
        "transforms_mode": "dct",
        "in_channels": 1,
        "checkpoint": "lightning_logs/version_0/checkpoints/epoch=8-step=228240.ckpt"
    },
    "dwt_resnet18": {
        "transforms_mode": "wavelet",
        "in_channels": 4,
        "checkpoint": "lightning_logs/version_0/checkpoints/epoch=15-step=405760.ckpt"
    },
    "dwt_rgb_resnet18": {
        "transforms_mode": "dwt_rgb",
        "in_channels": 7,
        "checkpoint": "lightning_logs/version_0/checkpoints/epoch=13-step=177520.ckpt"
    },
    "fft_magnitude_phase_resnet18": {
        "transforms_mode": "fft_mag_phase",
        "in_channels": 2,
        "checkpoint": "lightning_logs/version_0/checkpoints/epoch=8-step=228240.ckpt"
    },
    # "fft_magnitude_resnet18": {
    #     "in_channels": 1,
    #     "checkpoint": "lightning_logs/version_0/checkpoints/epoch=10-step=278960.ckpt"
    # },
    # "fft_phase_resnet18": {
    #     "in_channels": 1,
    #     "checkpoint": "lightning_logs/version_0/checkpoints/epoch=6-step=177520.ckpt"
    # },
    "rgb_resnet18": {
        "transforms_mode": "rgb",
        "in_channels": 3,
        "checkpoint": "lightning_logs/version_0/checkpoints/epoch=10-step=278960.ckpt"
    }
}