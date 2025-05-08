# import torch
# from datasets import load_dataset
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

# def extract_label(key_path):
#     """Extract label from the file path in '__key__'"""
#     if '/fake/' in key_path:
#         return 1  # Fake image
#     elif '/real/' in key_path:
#         return 0  # Real image
#     else:
#         # Additional checks for other path formats
#         path_lower = key_path.lower()
#         if 'fake' in path_lower:
#             return 1
#         elif 'real' in path_lower:
#             return 0
#         else:
#             return -1  # Unknown label

# def process_example(example):
#     """Process a single example and extract label"""
#     try:
#         # Extract label from key path
#         label = extract_label(example['__key__'])
        
#         # Return processed example with label
#         example['label'] = label
#         return example
            
#     except Exception as e:
#         print(f"Error processing example {example['__key__']}: {e}")
#         example['label'] = -1
#         return example

# def load_hf_dataset(dataset_path_or_name, max_samples=None, split_ratios=[0.7, 0.15, 0.15]):
#     """
#     Load a deepfake dataset from Hugging Face
    
#     Args:
#         dataset_path_or_name: Hugging Face dataset name
#         max_samples: specify subsample size of the dataset
#         split_ratios: Train/val/test split ratios
    
#     Returns:
#         train_dataset, val_dataset, test_dataset
#     """
#     try:
#         # First try loading as a Hugging Face dataset
#         print(f"Attempting to load dataset from Hugging Face: {dataset_path_or_name}")
#         is_streaming = max_samples is None
#         dataset = load_dataset(dataset_path_or_name, streaming=is_streaming)
#         print(f"Successfully loaded dataset from Hugging Face: {dataset_path_or_name}")

#         # Process to add labels
#         if 'train' in dataset:
#             # Apply the label extraction
#             if is_streaming:
#                 # For streaming datasets
#                 processed_dataset = dataset['train'].map(process_example)
                
#                 # Filter out examples with unknown labels - for streaming datasets
#                 processed_dataset = processed_dataset.filter(lambda x: x['label'] != -1)
                
#                 # Define train, val and test splits for streaming
#                 def get_split_datasets():
#                     counter = 0
#                     train_dataset = []
#                     val_dataset = []
#                     test_dataset = []
                    
#                     for example in processed_dataset:
#                         mod_val = counter % 100  # Use modulo for deterministic splitting
#                         if mod_val < int(split_ratios[0] * 100):  # Train split
#                             train_dataset.append(example)
#                         elif mod_val < int((split_ratios[0] + split_ratios[1]) * 100):  # Val split
#                             val_dataset.append(example)
#                         else:  # Test split
#                             test_dataset.append(example)
#                         counter += 1
                        
#                         # Break if max_samples is reached
#                         if max_samples and counter >= max_samples:
#                             break
                    
#                     return train_dataset, val_dataset, test_dataset
                
#                 # For streaming, we'll return functions that generate iterables
#                 return get_split_datasets
#             else:
#                 # For non-streaming datasets
#                 processed_dataset = dataset['train'].map(
#                     process_example,
#                     num_proc=2
#                 )
                
#                 # Filter out examples with unknown labels
#                 processed_dataset = processed_dataset.filter(lambda x: x['label'] != -1)
                
#                 # If max_samples is specified, select a subset
#                 if max_samples is not None:
#                     processed_dataset = processed_dataset.select(range(min(max_samples, len(processed_dataset))))
                
#                 # For in-memory datasets, use stratified splitting
#                 train_val = processed_dataset.train_test_split(
#                     test_size=(split_ratios[1] + split_ratios[2]), 
#                     stratify_by_column='label'
#                 )
#                 val_test = train_val['test'].train_test_split(
#                     test_size=split_ratios[2]/(split_ratios[1] + split_ratios[2]), 
#                     stratify_by_column='label'
#                 )
#                 train_dataset = train_val['train']
#                 val_dataset = val_test['train']
#                 test_dataset = val_test['test']
            
#                 return train_dataset, val_dataset, test_dataset
#         else:
#             raise ValueError("Dataset does not have expected 'train' split")
            
#     except Exception as e:
#         print(f"Could not load dataset. Error: {e}")
#         return None, None, None

# class EfficientDeepfakeDataset(Dataset):
#     """Memory-efficient dataset for deepfake detection using MobileNet"""
    
#     def __init__(self, dataset_items, transform=None):
#         self.dataset_items = dataset_items
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.dataset_items)
    
#     def __getitem__(self, idx):
#         # Get item from dataset
#         try:
#             example = self.dataset_items[idx]
            
#             # Get image and label
#             image = example['png']
#             label = example['label']
            
#             # Apply transforms if specified
#             if self.transform:
#                 image = self.transform(image)
            
#             return image, torch.tensor(label, dtype=torch.float32)
        
#         except Exception as e:
#             print(f"Error accessing item {idx}: {e}")
#             # Return a placeholder in case of error
#             if self.transform:
#                 placeholder = torch.zeros((3, 224, 224))
#             else:
#                 placeholder = Image.new('RGB', (224, 224), color='black')
#             return placeholder, torch.tensor(-1, dtype=torch.float32)

# class StreamingDeepfakeIterator:
#     """Iterator for streaming deepfake datasets"""
    
#     def __init__(self, stream_dataset, transform=None, max_samples=None):
#         self.stream_dataset = stream_dataset
#         self.transform = transform
#         self.max_samples = max_samples
#         self.counter = 0
    
#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         if self.max_samples and self.counter >= self.max_samples:
#             raise StopIteration
            
#         try:
#             example = next(self.stream_dataset)
#             self.counter += 1
            
#             # Get image and label
#             image = example['png']
#             label = example['label']
            
#             # Apply transforms if specified
#             if self.transform:
#                 image = self.transform(image)
                
#             return image, torch.tensor(label, dtype=torch.float32)
            
#         except StopIteration:
#             raise StopIteration
#         except Exception as e:
#             print(f"Error in streaming iterator: {e}")
#             # Return a placeholder in case of error
#             if self.transform:
#                 placeholder = torch.zeros((3, 224, 224))
#             else:
#                 placeholder = Image.new('RGB', (224, 224), color='black')
#             return placeholder, torch.tensor(-1, dtype=torch.float32)

# def create_efficient_loaders(dataset_source, batch_size=16, num_workers=4, max_samples=None):
#     """Create memory-efficient data loaders"""
    
#     # Define transformations - MobileNet specific
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                            std=[0.229, 0.224, 0.225])
#     ])
    
#     # Check if we're using streaming or not
#     if callable(dataset_source):  # For streaming datasets
#         # Get the split datasets
#         train_items, val_items, test_items = dataset_source()
        
#         # Create dataset objects
#         train_data = EfficientDeepfakeDataset(train_items, transform)
#         val_data = EfficientDeepfakeDataset(val_items, transform)
#         test_data = EfficientDeepfakeDataset(test_items, transform)
        
#     else:  # For non-streaming datasets
#         train_dataset, val_dataset, test_dataset = dataset_source
        
#         # Create dataset objects
#         train_data = EfficientDeepfakeDataset(train_dataset, transform)
#         val_data = EfficientDeepfakeDataset(val_dataset, transform)
#         test_data = EfficientDeepfakeDataset(test_dataset, transform)
    
#     # Create data loaders with prefetching and pinned memory
#     train_loader = DataLoader(
#         train_data,
#         batch_size=batch_size,
#         shuffle=True,  # We can shuffle now
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True
#     )
    
#     val_loader = DataLoader(
#         val_data,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )
    
#     test_loader = DataLoader(
#         test_data,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )
    
#     return train_loader, val_loader, test_loader   
import torch
from datasets import load_dataset, IterableDataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np

def extract_label(key_path):
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
        example['label'] = -1
        return example

def load_streaming_dataset(dataset_path_or_name, max_samples=None, seed=42):
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
        dataset = load_dataset(dataset_path_or_name, streaming=True)
        
        if 'train' not in dataset:
            raise ValueError(f"Dataset {dataset_path_or_name} does not have a 'train' split")
        
        # Process examples and add labels
        processed_dataset = dataset['train'].map(process_example)
        
        # Filter out examples with unknown labels
        filtered_dataset = processed_dataset.filter(lambda x: x['label'] != -1)
        
        # Shuffle the dataset with a buffer
        shuffled_dataset = filtered_dataset.shuffle(buffer_size=1000, seed=seed)
        
        # Create train, validation and test datasets using take/skip
        # For streaming datasets, we'll split using a counter approach
        def get_train_val_test_iterators():
            train_examples = []
            val_examples = []
            test_examples = []
            
            # Initialize counters
            count = 0
            
            # Create iterators
            dataset_iter = iter(shuffled_dataset)
            
            while True:
                try:
                    if max_samples and count >= max_samples:
                        break
                        
                    # Get next example
                    example = next(dataset_iter)
                    count += 1
                    
                    # Determine split based on counter
                    if count % 10 == 0:  # 10% for validation
                        val_examples.append(example)
                    elif count % 10 == 1:  # 10% for testing
                        test_examples.append(example)
                    else:  # 80% for training
                        train_examples.append(example)
                        
                    # Print progress
                    if count % 100 == 0:
                        print(f"Processed {count} examples: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test")
                        
                except StopIteration:
                    print("Reached end of dataset stream")
                    break
                except Exception as e:
                    print(f"Error processing example: {e}")
                    continue
            
            print(f"Finished processing. Total: {count}")
            print(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
            
            return train_examples, val_examples, test_examples
        
        return get_train_val_test_iterators
        
    except Exception as e:
        print(f"Error loading streaming dataset: {e}")
        return None

def load_regular_dataset(dataset_path_or_name, max_samples=None, seed=42):
    """
    Load a Hugging Face dataset in regular (non-streaming) mode
    
    Args:
        dataset_path_or_name: Hugging Face dataset name
        max_samples: Maximum number of samples to use
        seed: Random seed for splitting
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    try:
        print(f"Loading regular dataset: {dataset_path_or_name}")
        dataset = load_dataset(dataset_path_or_name, streaming=False)
        
        if 'train' not in dataset:
            raise ValueError(f"Dataset {dataset_path_or_name} does not have a 'train' split")
        
        # Process examples and add labels
        processed_dataset = dataset['train'].map(
            process_example,
            num_proc=2
        )
        
        # Filter out examples with unknown labels
        filtered_dataset = processed_dataset.filter(lambda x: x['label'] != -1)
        
        # Limit to max_samples if specified
        if max_samples and max_samples < len(filtered_dataset):
            dataset_subset = filtered_dataset.select(range(max_samples))
        else:
            dataset_subset = filtered_dataset
        
        # Split into train/val/test (70/15/15 by default)
        train_val = dataset_subset.train_test_split(test_size=0.3, seed=seed)
        val_test = train_val['test'].train_test_split(test_size=0.5, seed=seed)
        
        train_dataset = train_val['train']
        val_dataset = val_test['train']
        test_dataset = val_test['test']
        
        print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        print(f"Error loading regular dataset: {e}")
        return None, None, None

class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection"""
    
    def __init__(self, examples, transform=None):
        self.examples = examples
        self.transform = transform
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        try:
            example = self.examples[idx]
            
            # Get image and label
            image = example['png']
            label = example['label']
            
            # Apply transforms if specified
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.float32)
        
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            # Return a placeholder
            if self.transform:
                placeholder = torch.zeros((3, 224, 224))
            else:
                placeholder = Image.new('RGB', (224, 224), color='black')
            return placeholder, torch.tensor(-1, dtype=torch.float32)

def create_data_loaders(dataset_examples, batch_size=16, num_workers=2):
    """
    Create data loaders from dataset examples
    
    Args:
        dataset_examples: Either a tuple of (train_examples, val_examples, test_examples)
                          or a function that returns this tuple
        batch_size: Batch size for loaders
        num_workers: Number of worker processes
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Get examples
    if callable(dataset_examples):
        # For streaming datasets, call the function to get examples
        train_examples, val_examples, test_examples = dataset_examples()
    else:
        # For regular datasets, examples are already provided
        if isinstance(dataset_examples, (list, tuple)) and len(dataset_examples) == 3:
            train_dataset, val_dataset, test_dataset = dataset_examples
            
            # Convert datasets to lists of examples
            train_examples = [ex for ex in train_dataset]
            val_examples = [ex for ex in val_dataset]
            test_examples = [ex for ex in test_dataset]
        else:
            raise ValueError("dataset_examples must be a tuple of (train, val, test) datasets or a function that returns examples")
    
    # Create datasets
    train_dataset = DeepfakeDataset(train_examples, transform=transform)
    val_dataset = DeepfakeDataset(val_examples, transform=transform)
    test_dataset = DeepfakeDataset(test_examples, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def load_dataset_and_create_loaders(dataset_name, streaming=False, max_samples=None, batch_size=16, num_workers=2, seed=42):
    """
    Convenience function to load dataset and create loaders in one step
    
    Args:
        dataset_name: Hugging Face dataset name or path
        streaming: Whether to use streaming mode
        max_samples: Maximum number of samples to use
        batch_size: Batch size for loaders
        num_workers: Number of worker processes
        seed: Random seed
        
    Returns:
        train_loader, val_loader, test_loader
    """
    if streaming:
        # Use streaming approach
        dataset_func = load_streaming_dataset(dataset_name, max_samples, seed)
        return create_data_loaders(dataset_func, batch_size, num_workers)
    else:
        # Use regular approach
        datasets = load_regular_dataset(dataset_name, max_samples, seed)
        return create_data_loaders(datasets, batch_size, num_workers)

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load deepfake detection dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Hugging Face dataset name')
    parser.add_argument('--streaming', action='store_true', help='Use streaming mode')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker processes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset}")
    print(f"Mode: {'Streaming' if args.streaming else 'Regular'}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    
    # Load dataset and create loaders
    train_loader, val_loader, test_loader = load_dataset_and_create_loaders(
        args.dataset,
        streaming=args.streaming,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Print dataset information
    print("\nDataset information:")
    print(f"Train: {len(train_loader.dataset)} examples ({len(train_loader)} batches)")
    print(f"Val: {len(val_loader.dataset)} examples ({len(val_loader)} batches)")
    print(f"Test: {len(test_loader.dataset)} examples ({len(test_loader)} batches)")
    
    # Test by iterating through a few batches
    print("\nTesting train loader...")
    for i, (images, labels) in enumerate(train_loader):
        if i >= 3:  # Just test a few batches
            break
        print(f"Batch {i}: Images shape={images.shape}, Labels shape={labels.shape}")