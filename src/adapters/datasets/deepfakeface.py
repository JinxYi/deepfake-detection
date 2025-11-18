import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

DATASET_NAME = "OpenRL/DeepFakeFace"

def _process_sample(sample):
    try:
        sample["label"] = 1  # all data in DeepFakeFace huggingface are fake
        return sample
    except Exception as e:
        print(f"Error processing sample: {e}")
        sample["label"] = -1
        return sample


class DFFIterableDataset(torch.utils.data.IterableDataset):
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


class DFFDataModule(L.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=2, max_samples=None, seed=42, transforms=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.seed = seed
        self.transforms = transforms

    def setup(self, stage=None):
        dataset = load_dataset(DATASET_NAME, streaming=True)

        train_stream = dataset["train"].map(_process_sample).filter(lambda x: x['label'] != -1).shuffle(buffer_size=1000, seed=self.seed)
        val_stream   = dataset["validation"].map(_process_sample).filter(lambda x: x['label'] != -1)

        # wrap in iterable datasets
        self.train_dataset = DFFIterableDataset(train_stream, transform=self.transforms["train"], max_samples=self.max_samples)
        self.val_dataset   = DFFIterableDataset(val_stream, transform=self.transforms["val"], max_samples=self.max_samples)
        self.test_dataset  = DFFIterableDataset(val_stream, transform=self.transforms["test"], max_samples=self.max_samples)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)