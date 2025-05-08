import os
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# parent_dir = os.path.dirname(os.getcwd())
# dataset_dir = make_directory(parent_dir, "data")

# Load dataset from Hugging Face or local directory
def load_hf_dataset(dataset_path_or_name, max_samples=None, streaming=True, split_ratios=[0.7, 0.15, 0.15], seed=42):
    """
    Load a deepfake dataset either from Hugging Face
    
    Args:
        dataset_path_or_name: Hugging Face dataset name
        max_samples: specify subsample size of the dataset
        split_ratios: Train/val/test split ratios
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    try:
        # First try loading as a Hugging Face dataset
        print(f"Attempting to load dataset from Hugging Face: {dataset_path_or_name}")
        dataset = load_dataset(dataset_path_or_name, streaming=streaming)
        print(f"Successfully loaded dataset from Hugging Face: {dataset_path_or_name}")
        print("dataset", dataset)
        if max_samples:
            # Limit the number of samples in each split if specified
            dataset = dataset.map(lambda x: x[:max_samples], batched=True)
        train_dataset = dataset['train'].shuffle(buffer_size=1000, seed=seed)
        
        # Create validation stream by taking every 5th example
        def split_train_val(example, idx):
            return {'split': 'val' if idx % 5 == 0 else 'train'}
        
        split_dataset = train_dataset.map(split_train_val, with_indices=True)
        train_dataset = split_dataset.filter(lambda example: example['split'] == 'train')
        val_dataset = split_dataset
        return train_dataset, val_dataset, None
        # # Check if the dataset has predefined splits
        # if 'train' in dataset and 'validation' in dataset and 'test' in dataset:
        #     return dataset['train'], dataset['validation'], dataset['test']
        # elif 'train' in dataset and 'validation' in dataset:
        #     # Split validation into validation and test
        #     val_test = dataset['validation'].train_test_split(test_size=0.5)
        #     return dataset['train'], val_test['train'], val_test['test']
        # elif 'train' in dataset:
        #     # Create validation and test splits from train
        #     train_val = dataset['train'].train_test_split(test_size=(split_ratios[1] + split_ratios[2]))
        #     val_test = train_val['test'].train_test_split(test_size=split_ratios[2]/(split_ratios[1] + split_ratios[2]))
        #     return train_val['train'], val_test['train'], val_test['test']
        # else:
        #     # Single split, create train/val/test
        #     splits = dataset['train'].train_test_split(test_size=(split_ratios[1] + split_ratios[2]))
        #     val_test = splits['test'].train_test_split(test_size=split_ratios[2]/(split_ratios[1] + split_ratios[2]))
        #     return splits['train'], val_test['train'], val_test['test']
            
    except Exception as e:
        print(f"Could not load as a Hugging Face dataset. Error: {e}")


def extract_label(key_path):
    """Extract label from the file path in '__key__'"""
    # Based on your example: './152/fake/34/1512'
    # The label appears to be in the path (fake or real)
    if '/fake/' in key_path:
        return 1  # Fake image
    elif '/real/' in key_path:
        return 0  # Real image
    else:
        return -1  # Unknown label

def process_example(example):
    """Process a single example and extract label"""
    try:
        # Extract label from key path
        label = extract_label(example['__key__'])
        
        # Return processed example with label
        example['label'] = label
        return example
            
    except Exception as e:
        print(f"Error processing example {example['__key__']}: {e}")
        return None
    
def process_dataset(dataset):
    """
    Process the dataset by downloading images and preparing them for ML tasks.
    
    Args:
        dataset: A HuggingFace Dataset with 'png', '__key__', and '__url__' features
    
    Returns:
        Processed dataset with images as numpy arrays
    """
    # Apply processing to all examples in parallel
    processed_dataset = dataset.map(
        process_example,
        num_proc=2,  # Use multiple processes for speed
        # remove_columns=['png'],  # Remove original png column to save memory
        desc="Extracting labels from dataset"
    )
    
    # Print label distribution
    label_counts = processed_dataset.map(
        lambda x: {'count': 1}, 
        remove_columns=processed_dataset.column_names
    ).to_pandas().groupby('label').count()
    
    print(f"Dataset label distribution:")
    print(f"Real: {label_counts.get(0, 0)}")
    print(f"Fake: {label_counts.get(1, 0)}")
    
    return processed_dataset

# Prepare and process the dataset for the model
class DeepfakeDataset(Dataset):
    """Dataset class for deepfake detection"""
    def __init__(self, hf_dataset, image_processor, label_column="label", image_column="image"):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.label_column = label_column
        self.image_column = image_column
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        
        # Process the image
        inputs = self.image_processor(image, return_tensors="pt")
        # Remove batch dimension
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        
        # Get label
        label = torch.tensor(item[self.label_column], dtype=torch.float)
        
        return inputs, label

# Create data loaders
def create_data_loaders(train_dataset, val_dataset, test_dataset, image_processor, 
                        batch_size=16, num_workers=4):
    """Create PyTorch DataLoaders from datasets"""
    train_ds = DeepfakeDataset(train_dataset, image_processor["train"])
    val_ds = DeepfakeDataset(val_dataset, image_processor["val"])
    test_ds = DeepfakeDataset(test_dataset, image_processor["test"])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        # shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

def __main__():
    # Example usage
    dataset_name = "xingjunm/WildDeepfake"
    train, val, test = load_hf_dataset(dataset_name)

    print(f"Train dataset: {train}, \nValidation dataset: {val}, \nTest dataset: {test}")