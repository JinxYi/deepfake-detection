# import local config
import config

# import library dependencies
import numpy as np

# pytorch
import torch
import pytorch_lightning as L

# import local dependencies
# from src.adapters.datasets.wilddeepfake import WildDeepfakeDataModule
from src.adapters.datasets.sida import SidADataModule
from src.models.freqnet import LitFreqNet

if __name__ == "__main__":
    model_id = "frequency_freqnet"
    model_checkpoint_dir = f"{config.CHECKPOINTS_DIR}/{model_id}"

    from torchvision import transforms

    # --- common normalization (ImageNet) ---
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # --- training transform ---
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),        # resize frame
        transforms.Lambda(lambda img: img.convert("RGB")),  # force RGB
        transforms.RandomHorizontalFlip(),    # flip for augmentation
        transforms.ColorJitter(               # optional: color variation
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        imagenet_normalize,
    ])

    # --- validation transform ---
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # deterministic resize
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        imagenet_normalize,
    ])

    # --- test transform (usually same as val) ---
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        imagenet_normalize,
    ])

    transforms = {
        "train": train_transform,
        "val": val_transform,
        "test": test_transform
    }

    # Set seeds for reproducibility
    seed = config.SEED

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_name = "xingjunm/WildDeepfake"
    max_samples = 994_000  # For quick development, remove for full dataset
    batch_size = 16
    num_workers = 0
    max_epochs = 20

    from src.adapters.datasets.wilddeepfake import load_streaming_dataset, create_data_loaders
    datasets = load_streaming_dataset(
        dataset_name,
        max_samples=max_samples,
        seed=seed
    )
    train_loader, val_loader, test_loader = create_data_loaders(
        datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        transforms=transforms,
        # additional_transforms=ela
    )

    # define early stopper
    early_stop_callback = L.callbacks.EarlyStopping(
        monitor="val_loss",       # metric to track
        patience=3,               # epochs to wait for improvement
        mode="min",               # "min" because we want val_loss to decrease
        verbose=True
    )

    # define ligntning checkpoint
    best_loss_checkpoint = L.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # define model
    deepfake_detector = LitFreqNet()

    trainer = L.Trainer(
        devices=1,
        callbacks=[early_stop_callback, best_loss_checkpoint],
        default_root_dir=model_checkpoint_dir,
        log_every_n_steps=10,
        profiler="simple", # track time taken
        max_steps= max_epochs * max_samples / batch_size, #(desired_epochs × dataset_size) / batch_size
        # limit_train_batches=1000,   # how many batches per "epoch"
        # limit_val_batches=200,      # how many val batches per "epoch"
    )

    # train model
    # trainer.fit(deepfake_detector, datamodule=sida_data_module)
    trainer.fit(deepfake_detector, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # test dataset on unseen samples
    # trainer.test(deepfake_detector, datamodule=sida_data_module)
    trainer.test(deepfake_detector, test_loader)