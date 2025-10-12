from collections.abc import Callable
import torch
from datasets import load_dataset, IterableDataset
from torch.utils.data import DataLoader, Dataset
import config
import pytorch_lightning as L
from PIL import Image
import io
from torchvision.transforms.functional import to_pil_image

DATASET_NAME = "xingjunm/WildDeepfake"

def dict_to_pil(image_dict):
    if isinstance(image_dict, dict) and "bytes" in image_dict:
        return Image.open(io.BytesIO(image_dict["bytes"]))
    raise ValueError(f"Cannot convert to PIL: {type(image_dict)}")

def _extract_label(key_path):
    """Extract label from the file path in '__key__'"""
    if '/fake/' in key_path:
        return 1  # Fake image
    elif '/real/' in key_path:
        return 0  # Real image
    else:
        # Additional checks for other path formats
        path_lower = key_path.lower()
        if 'fake' in path_lower:
            return 1
        elif 'real' in path_lower:
            return 0
        else:
            return -1  # Unknown label

def _process_sample(sample):
    """Process a single sample and extract label"""
    try:
        # Return processed sample with label
        sample['label'] = _extract_label(sample['__key__'])
        return sample
            
    except Exception as e:
        print(f"Error processing sample {sample['__key__']}: {e}")
        sample['label'] = -1
        return sample

def load_streaming_dataset(dataset_path_or_name=DATASET_NAME, max_samples=None, seed=config.SEED):
    """
    Load a Hugging Face dataset in streaming mode
    
    Args:
        dataset_path_or_name: Hugging Face dataset name
        max_samples: Maximum number of samples to use (only as a counter limit)
        seed: Random seed for shuffling
        
    Returns:
        train_dataset, val_dataset, test_dataset as streaming iterators
    """
    try:
        print(f"Loading streaming dataset: {dataset_path_or_name}")
        dataset = load_dataset(dataset_path_or_name, streaming=True, split="train")
        
        if 'train' not in dataset:
            raise ValueError(f"Dataset {dataset_path_or_name} does not have a 'train' split")
        
        # Process samples and add labels
        train_dataset = dataset['train'].map(_process_sample)
        # test_dataset = dataset['test'].map(_process_sample)
        
        # # Shuffle the dataset with a buffer
        # shuffled_train_dataset = train_dataset.shuffle(buffer_size=config.STREAM_SHUFFLE_BUFFER_SIZE, seed=seed)
        
        # # Create train, validation and test datasets using take/skip
        # # For streaming datasets, we'll split using a counter approach
        # def split_train_val(ds, max_samples=None):
        #     def gen(split):
        #         count = 0
        #         for sample in ds:
        #             if max_samples and count >= max_samples:
        #                 break
        #             count += 1
        #             if count % 10 == 0 and split == "val":  # ~10% validation
        #                 yield sample
        #             elif count % 10 != 0 and split == "train":
        #                 yield sample
        #     return gen("train"), gen("val")

        # train_iter, val_iter = split_train_val(shuffled_train_dataset, max_samples=max_samples)

        return train_dataset
        
    except Exception as e:
        print(f"Error loading streaming dataset: {e}")
        raise

class WildDeepfakeIterableDataset(IterableDataset):
    """Dataset for deepfake detection"""
    
    def __init__(self, samples, transform=None, additional_transforms=None, max_samples=None):
        self.dataset = samples
        self.transform = transform
        self.additional_transforms = additional_transforms
        self.max_samples = max_samples
    
    def __iter__(self):
        count = 0
        for sample in self.dataset:
            if self.max_samples and count >= self.max_samples:
                break
            count += 1

            try:
                image = sample["png"]
                if isinstance(image, dict):
                    image = dict_to_pil(image)
                label = sample["label"]

                # Perform transformations
                if self.additional_transforms:
                    image = self.additional_transforms(image)

                if self.transform:
                    image = self.transform(image)

                yield image, torch.tensor(label, dtype=torch.float32)
            except Exception as e:
                print(f"Error loading sample {count}: {e}")
                placeholder = torch.zeros((3, 224, 224))  # replace 224 with your `image_size`
                raise
                # yield placeholder, torch.tensor(-1, dtype=torch.float32)
    def __len__(self):
        if self.max_samples:
            return self.max_samples
        return 900000
    
def create_data_loaders(datasets: tuple[list, list, list], batch_size=16, num_workers=2, transforms = None, additional_transforms: Callable | None = None):
    """
    Create data loaders from dataset
    
    Args:
        datasets: Either a tuple of (train_samples, val_samples, test_samples)
                          or a function that returns this tuple
        batch_size: Batch size for loaders
        num_workers: Number of worker processes
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get samples
    if isinstance(datasets, (list, tuple)) and len(datasets) == 3:
        train_samples, val_samples, test_samples = datasets
    else:
        raise ValueError("datasets must be a tuple of (train, val, test) datasets or a function that returns samples")
    
    # Create datasets
    train_dataset = WildDeepfakeIterableDataset(train_samples, transform=transforms["train"], additional_transforms=additional_transforms)
    val_dataset = WildDeepfakeIterableDataset(val_samples, transform=transforms["val"], additional_transforms=additional_transforms)
    test_dataset = WildDeepfakeIterableDataset(test_samples, transform=transforms["test"], additional_transforms=additional_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

class WildDeepfakeDataset(Dataset):
    """Map-style Dataset for deepfake detection (non-streaming)"""

    def __init__(self, samples, transform=None, additional_transforms=None):
        self.samples = samples  # Ensure it's indexable
        self.transform = transform
        self.additional_transforms = additional_transforms

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # sample = _process_sample(sample)
        try:
            image = sample["png"]
            label = sample["label"]

            if self.additional_transforms:
                image = self.additional_transforms(image)
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            placeholder = torch.zeros((3, 224, 224))  # Adjust image size as needed
            return placeholder, torch.tensor(-1, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)
    
class WildDeepfakeDataModule(L.LightningDataModule):
    def __init__(self, dataset_name=DATASET_NAME, batch_size=32, num_workers=2, max_samples=None, seed=42, transforms: dict = None, additional_transforms=None, dataset_cache_dir=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.seed = seed
        self.transforms = transforms
        self.additional_transforms = additional_transforms
        self.dataset_cache_dir = dataset_cache_dir

    def setup(self, stage=None):
        """Load and preprocess datasets (called on every process in DDP)."""
        if self.dataset_cache_dir:
            dataset = load_dataset(self.dataset_name, cache_dir=self.dataset_cache_dir)
        else:
            dataset = load_dataset(self.dataset_name)

        # Preprocess and filter
        # train_ds = dataset["train"].shuffle(seed=self.seed)
        # test_ds  = dataset["test"]
        print("Dataset loaded. Processing samples...")
        train_ds = dataset["train"].map(_process_sample, num_proc=self.num_workers)
        test_ds  = dataset["test"].map(_process_sample, num_proc=self.num_workers)
        
        print("Extracted labels. Generating train/val split...")
        split = train_ds.train_test_split(test_size=0.2, seed=self.seed)
        train_ds = split["train"]
        val_ds = split["test"]
        # train_ds, val_ds = train_ds.train_test_split(test_size=0.2, seed=self.seed) # create validation set

        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
        # Wrap with your iterable dataset class
        self.train_dataset = WildDeepfakeDataset(train_ds, transform=self.transforms["train"], additional_transforms=self.additional_transforms)
        self.val_dataset   = WildDeepfakeDataset(val_ds, transform=self.transforms["val"], additional_transforms=self.additional_transforms)
        self.test_dataset  = WildDeepfakeDataset(test_ds, transform=self.transforms["test"], additional_transforms=self.additional_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False
        )