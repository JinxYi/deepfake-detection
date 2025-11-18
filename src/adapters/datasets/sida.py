import torch
from datasets import load_dataset, IterableDataset
from torch.utils.data import DataLoader
import pytorch_lightning as L
from PIL import Image

DATASET_NAME = "saberzl/SID_Set"

# class SidAIterableDataset(IterableDataset):
#     """Dataset for deepfake detection"""
    
#     def __init__(self, samples, transform=None, additional_transforms=None, max_samples=None):
#         self.dataset = samples
#         self.transform = transform
#         self.additional_transforms = additional_transforms
#         self.max_samples = max_samples
    
#     def __iter__(self):
#         count = 0
#         for sample in self.dataset:
#             if self.max_samples and count >= self.max_samples:
#                 break
#             count += 1

#             try:
#                 image = sample["image"]
#                 label = sample["label"]

#                 # Perform transformations
#                 if self.additional_transforms:
#                     image = self.additional_transforms(image)

#                 if self.transform:
#                     image = self.transform(image)

#                 yield image, torch.tensor(label, dtype=torch.float32)
#             except Exception as e:
#                 print(f"Error loading sample {count}: {e}")
#                 placeholder = torch.zeros((3, 224, 224))  # replace 224 with your `image_size`
#                 yield placeholder, torch.tensor(-1, dtype=torch.float32)
#     def __len__(self):
#         if self.max_samples:
#             return self.max_samples
#         return 10000
    
# def _process_sample(sample):
#     """Process a single sample and extract label"""
#     try:
#         # Return processed sample with label
#         sample['label'] = 0 if sample['label'] == 0 else 1
#         return sample
            
#     except Exception as e:
#         print(f"Error processing sample {sample['__key__']}: {e}")
#         sample['label'] = -1
#         return sample
    
# class SidADataModule(L.LightningDataModule):
#     def __init__(self, batch_size=16, num_workers=0, max_samples=None, seed=42, transforms: dict = None, additional_transforms=None):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.max_samples = max_samples
#         self.seed = seed
#         self.transforms = transforms
#         self.additional_transforms = additional_transforms

#     def setup(self, stage=None):
#         """Load and preprocess datasets (called on every process in DDP)."""
#         dataset = load_dataset(DATASET_NAME, streaming=True)

#         # Preprocess and filter
#         train_ds = dataset["train"].map(_process_sample).filter(lambda x: x['label'] != -1).shuffle(buffer_size=5000, seed=self.seed)
#         test_ds  = dataset["validation"].map(_process_sample).filter(lambda x: x['label'] != -1)

#         def split_train_val(ds, val_size=0.1, max_samples=None):
#             if val_size < 0 or val_size > 1:
#                 raise ValueError("Valication split size should be between 0 and 1")
#             partition = val_size * 10
#             def gen(split):
#                 count = 0
#                 for sample in ds:
#                     if max_samples and count >= max_samples:
#                         break
#                     count += 1
#                     if count % partition == 0 and split == "val":  # ~ partition % validation
#                         yield sample
#                     elif count % partition != 0 and split == "train":
#                         yield sample
#             return gen("train"), gen("val")

#         train_ds, val_ds = split_train_val(train_ds, max_samples=self.max_samples)

#         # Wrap with your iterable dataset class
#         self.train_dataset = SidAIterableDataset(train_ds, transform=self.transforms["train"], additional_transforms=self.additional_transforms, max_samples=self.max_samples)
#         self.val_dataset   = SidAIterableDataset(val_ds, transform=self.transforms["val"], additional_transforms=self.additional_transforms, max_samples=self.max_samples)
#         self.test_dataset  = SidAIterableDataset(test_ds, transform=self.transforms["test"], additional_transforms=self.additional_transforms, max_samples=self.max_samples)

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,   # can't shuffle IterableDataset
#             num_workers=self.num_workers,
#             pin_memory=False,
#             drop_last=True
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=False
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=False
#         )
def _process_sample(sample):
    try:
        sample["label"] = int(0 if sample['label'] == 0 else 1)  # ensure int
        return sample
    except Exception as e:
        print(f"Error processing sample: {e}")
        sample["label"] = -1
        return sample


class SidAIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator, transform=None, max_samples=None):
        super().__init__()
        self.generator = generator
        self.transform = transform
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        for sample in self.generator:
            if self.max_samples and count >= self.max_samples:
                break
            count += 1

            try:
                img = sample["image"]
                # force to RGB
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                yield img, torch.tensor(sample["label"], dtype=torch.float32)
            except Exception as e:
                print(f"Error loading sample {count}: {e}")
                yield torch.zeros((3, 256, 256)), torch.tensor(-1.0)

    def __len__(self):
        if self.max_samples:
            return self.max_samples
        raise TypeError("IterableDataset has no length")


class SidADataModule(L.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=2, max_samples=None, seed=42, transforms=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.seed = seed
        self.transforms = transforms

    def setup(self, stage=None):
        dataset = load_dataset(DATASET_NAME, streaming=True, split="validation")
        val_stream   = dataset.map(_process_sample)
        # train_stream = dataset["train"].map(_process_sample).filter(lambda x: x['label'] != -1).shuffle(buffer_size=1000, seed=self.seed)
        # val_stream   = dataset["validation"].map(_process_sample).filter(lambda x: x['label'] != -1)

        # wrap in iterable datasets
        # self.train_dataset = SidAIterableDataset(train_stream, transform=self.transforms["train"], max_samples=self.max_samples)
        # self.val_dataset   = SidAIterableDataset(val_stream, transform=self.transforms["val"], max_samples=self.max_samples)
        self.test_dataset  = SidAIterableDataset(val_stream, transform=self.transforms["test"], max_samples=self.max_samples)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)