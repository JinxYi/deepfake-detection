import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset
import torch
import cv2
import numpy as np
from datasets import load_dataset, IterableDataset as HFIterableDataset
from transformers import AutoImageProcessor
import torchvision.transforms as transforms
from typing import Optional, Dict, Any, List, Tuple
import io
import base64
from PIL import Image
import random
import warnings
warnings.filterwarnings("ignore")


class VideoStreamingDataset(IterableDataset):
    """
    Streaming dataset for video deepfake detection using HuggingFace datasets
    Handles both local video files and streaming datasets
    """
    
    def __init__(
        self,
        dataset_name: str = None,
        dataset_config: str = None,
        split: str = "train",
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        frame_sampling: str = "uniform",  # "uniform", "random", "consecutive"
        streaming: bool = True,
        max_samples: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_sampling = frame_sampling
        self.streaming = streaming
        self.max_samples = max_samples
        self.transform = transform or self._default_transforms()
        self.cache_dir = cache_dir
        
        # Load dataset
        self.dataset = self._load_dataset()
        
    def _load_dataset(self):
        """Load HuggingFace dataset"""
        try:
            if self.dataset_name:
                dataset = load_dataset(
                    self.dataset_name,
                    self.dataset_config,
                    split=self.split,
                    streaming=self.streaming,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                )
            else:
                # Fallback for local datasets or custom loading
                raise ValueError("Please provide dataset_name for HuggingFace datasets")
                
            if self.max_samples and self.streaming:
                dataset = dataset.take(self.max_samples)
                
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Return a dummy dataset for testing
            return self._create_dummy_dataset()
    
    def _create_dummy_dataset(self):
        """Create dummy dataset for testing purposes"""
        def dummy_generator():
            for i in range(100):
                # Create dummy video data
                frames = np.random.randint(0, 255, (self.num_frames, *self.frame_size, 3), dtype=np.uint8)
                label = random.randint(0, 1)
                
                yield {
                    "video": frames,
                    "label": label,
                    "video_id": f"dummy_video_{i}"
                }
        
        return dummy_generator()
    
    def _default_transforms(self):
        """Default image transforms"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _decode_video_bytes(self, video_bytes: bytes) -> np.ndarray:
        """Decode video from bytes"""
        try:
            # Save bytes to temporary file
            temp_path = "/tmp/temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_bytes)
            
            # Read video using OpenCV
            cap = cv2.VideoCapture(temp_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            return np.array(frames)
            
        except Exception as e:
            print(f"Error decoding video: {e}")
            # Return dummy frames
            return np.random.randint(0, 255, (30, *self.frame_size, 3), dtype=np.uint8)
    
    def _decode_video_path(self, video_path: str) -> np.ndarray:
        """Decode video from file path"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            return np.array(frames)
            
        except Exception as e:
            print(f"Error reading video from path: {e}")
            return np.random.randint(0, 255, (30, *self.frame_size, 3), dtype=np.uint8)
    
    def _sample_frames(self, frames: np.ndarray) -> np.ndarray:
        """Sample frames from video according to sampling strategy"""
        total_frames = len(frames)
        
        if total_frames < self.num_frames:
            # Repeat frames if not enough
            indices = np.tile(np.arange(total_frames), 
                            (self.num_frames // total_frames + 1))[:self.num_frames]
        else:
            if self.frame_sampling == "uniform":
                # Uniformly sample frames
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            elif self.frame_sampling == "random":
                # Randomly sample frames
                indices = np.sort(np.random.choice(total_frames, self.num_frames, replace=False))
            elif self.frame_sampling == "consecutive":
                # Sample consecutive frames from random start
                start_idx = np.random.randint(0, max(1, total_frames - self.num_frames + 1))
                indices = np.arange(start_idx, start_idx + self.num_frames)
            else:
                raise ValueError(f"Unknown sampling method: {self.frame_sampling}")
        
        return frames[indices]
    
    def _process_sample(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single sample from the dataset"""
        try:
            # Extract video data
            if "video" in sample:
                if isinstance(sample["video"], bytes):
                    frames = self._decode_video_bytes(sample["video"])
                elif isinstance(sample["video"], str):
                    frames = self._decode_video_path(sample["video"])
                elif isinstance(sample["video"], np.ndarray):
                    frames = sample["video"]
                else:
                    # Handle other formats (PIL Images, etc.)
                    frames = np.array(sample["video"])
            else:
                # Handle datasets with different video key names
                video_keys = ["video_path", "path", "file", "frames"]
                frames = None
                for key in video_keys:
                    if key in sample:
                        if isinstance(sample[key], str):
                            frames = self._decode_video_path(sample[key])
                        else:
                            frames = np.array(sample[key])
                        break
                
                if frames is None:
                    raise ValueError("No video data found in sample")
            
            # Sample frames
            sampled_frames = self._sample_frames(frames)
            
            # Apply transforms to each frame
            transformed_frames = []
            for frame in sampled_frames:
                if self.transform:
                    frame_tensor = self.transform(frame)
                else:
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                transformed_frames.append(frame_tensor)
            
            # Stack frames: (num_frames, channels, height, width)
            video_tensor = torch.stack(transformed_frames, dim=0)
            
            # Extract label
            if "label" in sample:
                label = sample["label"]
            elif "is_fake" in sample:
                label = int(sample["is_fake"])
            elif "target" in sample:
                label = sample["target"]
            else:
                # Default label for testing
                label = 0
            
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return video_tensor, label_tensor
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            # Return dummy data
            dummy_frames = torch.randn(self.num_frames, 3, *self.frame_size)
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_frames, dummy_label
    
    def __iter__(self):
        """Iterate over the dataset"""
        for sample in self.dataset:
            try:
                video_tensor, label_tensor = self._process_sample(sample)
                yield video_tensor, label_tensor
            except Exception as e:
                print(f"Error in dataset iteration: {e}")
                continue


class MAVOSDataModule(pl.LightningDataModule):
    """
    Specialized PyTorch Lightning DataModule for MAVOS-DD dataset
    Handles multilingual audio-video deepfake detection with open-set evaluation
    """
    
    def __init__(
        self,
        dataset_name: str = "unibuc-cs/MAVOS-DD",
        
        # Video processing parameters
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        frame_sampling: str = "uniform",
        
        # DataLoader parameters
        batch_size: int = 4,
        num_workers: int = 2,
        
        # MAVOS-DD specific parameters
        language_filter: Optional[str] = None,  # Filter by language
        open_set_mode: str = "indomain",  # "indomain", "open_model", "open_language", "open_full"
        
        # Streaming parameters
        streaming: bool = True,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        
        # Other parameters
        cache_dir: Optional[str] = None,
        pin_memory: bool = True,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.dataset_name = dataset_name
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_sampling = frame_sampling
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.streaming = streaming
        
        # MAVOS-DD specific settings
        self.language_filter = language_filter
        self.open_set_mode = open_set_mode
        
        # Sample limits
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        
        self.cache_dir = cache_dir
        
        # Transforms
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
    
    def _get_train_transforms(self):
        """Training transforms with augmentation"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_val_transforms(self):
        """Validation/test transforms without augmentation"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _filter_dataset(self, dataset, split_name: str):
        """Filter dataset based on MAVOS-DD specific criteria"""
        
        def filter_function(sample):
            # Filter by split
            if sample["split"] != split_name:
                return False
            
            # Filter by language if specified
            if self.language_filter and sample["language"] != self.language_filter:
                return False
            
            # Filter by open-set mode for test split
            if split_name == "test":
                if self.open_set_mode == "indomain":
                    return not sample["open_set_model"] and not sample["open_set_language"]
                elif self.open_set_mode == "open_model":
                    return sample["open_set_model"] and not sample["open_set_language"]
                elif self.open_set_mode == "open_language":
                    return not sample["open_set_model"] and sample["open_set_language"]
                elif self.open_set_mode == "open_full":
                    return True  # Include all test samples
            
            return True
        
        return dataset.filter(filter_function)
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages"""
        
        if stage == "fit" or stage is None:
            # Load full dataset
            full_dataset = load_dataset(
                self.dataset_name,
                streaming=self.streaming,
                cache_dir=self.cache_dir
            )
            
            # Training dataset
            train_dataset = self._filter_dataset(full_dataset["train"], "train")
            if self.max_train_samples and self.streaming:
                train_dataset = train_dataset.take(self.max_train_samples)
            
            self.train_dataset_obj = MAVOSStreamingDataset(
                dataset=train_dataset,
                num_frames=self.num_frames,
                frame_size=self.frame_size,
                frame_sampling=self.frame_sampling,
                transform=self.train_transform,
            )
            
            # Validation dataset
            val_dataset = self._filter_dataset(full_dataset["train"], "validation")
            if self.max_val_samples and self.streaming:
                val_dataset = val_dataset.take(self.max_val_samples)
            
            self.val_dataset_obj = MAVOSStreamingDataset(
                dataset=val_dataset,
                num_frames=self.num_frames,
                frame_size=self.frame_size,
                frame_sampling="uniform",  # Always uniform for validation
                transform=self.val_transform,
            )
        
        if stage == "test" or stage is None:
            # Load test dataset
            full_dataset = load_dataset(
                self.dataset_name,
                streaming=self.streaming,
                cache_dir=self.cache_dir
            )
            
            test_dataset = self._filter_dataset(full_dataset["train"], "test")
            if self.max_test_samples and self.streaming:
                test_dataset = test_dataset.take(self.max_test_samples)
            
            self.test_dataset_obj = MAVOSStreamingDataset(
                dataset=test_dataset,
                num_frames=self.num_frames,
                frame_size=self.frame_size,
                frame_sampling="uniform",
                transform=self.val_transform,
            )
    
    def train_dataloader(self):
        """Training dataloader"""
        return DataLoader(
            self.train_dataset_obj,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self):
        """Validation dataloader"""
        return DataLoader(
            self.val_dataset_obj,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
    
    def test_dataloader(self):
        """Test dataloader"""
        return DataLoader(
            self.test_dataset_obj,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


class MAVOSStreamingDataset(IterableDataset):
    """
    Specialized streaming dataset for MAVOS-DD
    Handles the specific structure and metadata of MAVOS-DD dataset
    """
    
    def __init__(
        self,
        dataset,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        frame_sampling: str = "uniform",
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_sampling = frame_sampling
        self.transform = transform
    
    def _decode_video_from_sample(self, sample: Dict[str, Any]) -> np.ndarray:
        """Decode video from MAVOS-DD sample"""
        try:
            # MAVOS-DD stores videos in the 'video' field
            video_data = sample["video"]
            
            if isinstance(video_data, bytes):
                # Video stored as bytes
                return self._decode_video_bytes(video_data)
            elif hasattr(video_data, 'read'):
                # Video stored as file-like object
                video_bytes = video_data.read()
                return self._decode_video_bytes(video_bytes)
            else:
                # Handle other formats
                print(f"Unknown video format: {type(video_data)}")
                return self._generate_dummy_frames()
                
        except Exception as e:
            print(f"Error decoding MAVOS-DD video: {e}")
            return self._generate_dummy_frames()
    
    def _decode_video_bytes(self, video_bytes: bytes) -> np.ndarray:
        """Decode video from bytes"""
        try:
            import tempfile
            import os
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(video_bytes)
                temp_path = temp_file.name
            
            # Read video using OpenCV
            cap = cv2.VideoCapture(temp_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return np.array(frames)
            
        except Exception as e:
            print(f"Error decoding video bytes: {e}")
            return self._generate_dummy_frames()
    
    def _generate_dummy_frames(self) -> np.ndarray:
        """Generate dummy frames when video loading fails"""
        return np.random.randint(0, 255, (self.num_frames, *self.frame_size, 3), dtype=np.uint8)
    
    def _sample_frames(self, frames: np.ndarray) -> np.ndarray:
        """Sample frames from video according to sampling strategy"""
        total_frames = len(frames)
        
        if total_frames < self.num_frames:
            # Repeat frames if not enough
            indices = np.tile(np.arange(total_frames), 
                            (self.num_frames // total_frames + 1))[:self.num_frames]
        else:
            if self.frame_sampling == "uniform":
                # Uniformly sample frames
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            elif self.frame_sampling == "random":
                # Randomly sample frames
                indices = np.sort(np.random.choice(total_frames, self.num_frames, replace=False))
            elif self.frame_sampling == "consecutive":
                # Sample consecutive frames from random start
                start_idx = np.random.randint(0, max(1, total_frames - self.num_frames + 1))
                indices = np.arange(start_idx, start_idx + self.num_frames)
            else:
                raise ValueError(f"Unknown sampling method: {self.frame_sampling}")
        
        return frames[indices]
    
    def _process_sample(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Process a single MAVOS-DD sample"""
        try:
            # Decode video
            frames = self._decode_video_from_sample(sample)
            
            # Sample frames
            sampled_frames = self._sample_frames(frames)
            
            # Apply transforms to each frame
            transformed_frames = []
            for frame in sampled_frames:
                if self.transform:
                    frame_tensor = self.transform(frame)
                else:
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                transformed_frames.append(frame_tensor)
            
            # Stack frames: (num_frames, channels, height, width)
            video_tensor = torch.stack(transformed_frames, dim=0)
            
            # Extract label - MAVOS-DD uses "fake" and "real" strings
            label_str = sample["label"]
            label = 1 if label_str == "fake" else 0
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            # Extract metadata
            metadata = {
                "language": sample.get("language", "unknown"),
                "generative_method": sample.get("generative_method", "unknown"),
                "video_path": sample.get("video_path", "unknown"),
                "open_set_model": sample.get("open_set_model", False),
                "open_set_language": sample.get("open_set_language", False),
            }
            
            return video_tensor, label_tensor, metadata
            
        except Exception as e:
            print(f"Error processing MAVOS-DD sample: {e}")
            # Return dummy data
            dummy_frames = torch.randn(self.num_frames, 3, *self.frame_size)
            dummy_label = torch.tensor(0, dtype=torch.long)
            dummy_metadata = {"language": "unknown", "generative_method": "unknown"}
            return dummy_frames, dummy_label, dummy_metadata
    
    def __iter__(self):
        """Iterate over the dataset"""
        for sample in self.dataset:
            try:
                video_tensor, label_tensor, metadata = self._process_sample(sample)
                yield video_tensor, label_tensor  # Only return video and label for standard training
            except Exception as e:
                print(f"Error in MAVOS dataset iteration: {e}")
                continue


class VideoDeepfakeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for video deepfake detection
    Supports HuggingFace streaming datasets and local video datasets
    """
    
    def __init__(
        self,
        # Dataset parameters
        train_dataset: str = None,
        val_dataset: str = None,
        test_dataset: str = None,
        dataset_config: str = None,
        
        # Video processing parameters
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        frame_sampling: str = "uniform",
        
        # DataLoader parameters
        batch_size: int = 4,
        num_workers: int = 2,
        
        # Streaming parameters
        streaming: bool = True,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        
        # Other parameters
        cache_dir: Optional[str] = None,
        pin_memory: bool = True,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Dataset names
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset or train_dataset
        self.test_dataset = test_dataset or train_dataset
        self.dataset_config = dataset_config
        
        # Video processing
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_sampling = frame_sampling
        
        # DataLoader settings
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Streaming settings
        self.streaming = streaming
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        
        self.cache_dir = cache_dir
        
        # Transforms
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
    def _get_train_transforms(self):
        """Training transforms with augmentation"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_val_transforms(self):
        """Validation/test transforms without augmentation"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages"""
        
        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset_obj = VideoStreamingDataset(
                dataset_name=self.train_dataset,
                dataset_config=self.dataset_config,
                split="train",
                num_frames=self.num_frames,
                frame_size=self.frame_size,
                frame_sampling=self.frame_sampling,
                streaming=self.streaming,
                max_samples=self.max_train_samples,
                transform=self.train_transform,
                cache_dir=self.cache_dir,
            )
            
            # Validation dataset
            self.val_dataset_obj = VideoStreamingDataset(
                dataset_name=self.val_dataset,
                dataset_config=self.dataset_config,
                split="validation",
                num_frames=self.num_frames,
                frame_size=self.frame_size,
                frame_sampling="uniform",  # Always uniform for validation
                streaming=self.streaming,
                max_samples=self.max_val_samples,
                transform=self.val_transform,
                cache_dir=self.cache_dir,
            )
        
        if stage == "test" or stage is None:
            # Test dataset
            self.test_dataset_obj = VideoStreamingDataset(
                dataset_name=self.test_dataset,
                dataset_config=self.dataset_config,
                split="test",
                num_frames=self.num_frames,
                frame_size=self.frame_size,
                frame_sampling="uniform",
                streaming=self.streaming,
                max_samples=self.max_test_samples,
                transform=self.val_transform,
                cache_dir=self.cache_dir,
            )
    
    def train_dataloader(self):
        """Training dataloader"""
        return DataLoader(
            self.train_dataset_obj,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # Important for consistent batch sizes
        )
    
    def val_dataloader(self):
        """Validation dataloader"""
        return DataLoader(
            self.val_dataset_obj,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
    
    def test_dataloader(self):
        """Test dataloader"""
        return DataLoader(
            self.test_dataset_obj,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


# Example usage and dataset configurations
def create_mavos_dd_datamodule(
    batch_size: int = 4,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    streaming: bool = True,
    max_samples: Optional[int] = None,
    language_filter: Optional[str] = None,  # e.g., "english", "arabic"
    open_set_mode: str = "indomain"  # "indomain", "open_model", "open_language", "open_full"
):
    """Create DataModule for MAVOS-DD dataset"""
    
    return MAVOSDataModule(
        dataset_name="unibuc-cs/MAVOS-DD",
        num_frames=num_frames,
        frame_size=frame_size,
        batch_size=batch_size,
        streaming=streaming,
        max_train_samples=max_samples,
        max_val_samples=max_samples // 5 if max_samples else None,
        num_workers=2,
        language_filter=language_filter,
        open_set_mode=open_set_mode,
    )


# def create_faceforensics_datamodule(
#     batch_size: int = 4,
#     num_frames: int = 16,
#     frame_size: Tuple[int, int] = (224, 224),
#     streaming: bool = True,
#     max_samples: Optional[int] = None
# ):
#     """Create DataModule for FaceForensics++ dataset"""
    
#     return VideoDeepfakeDataModule(
#         train_dataset="face_forensics_plus_plus",  # Example dataset name
#         dataset_config="raw",  # or "c23", "c40" for different compression levels
#         num_frames=num_frames,
#         frame_size=frame_size,
#         batch_size=batch_size,
#         streaming=streaming,
#         max_train_samples=max_samples,
#         max_val_samples=max_samples // 5 if max_samples else None,
#         num_workers=2,
#     )


# def create_celebdf_datamodule(
#     batch_size: int = 4,
#     num_frames: int = 16,
#     frame_size: Tuple[int, int] = (224, 224),
#     streaming: bool = True
# ):
#     """Create DataModule for Celeb-DF dataset"""
    
#     return VideoDeepfakeDataModule(
#         train_dataset="celeb_df",
#         num_frames=num_frames,
#         frame_size=frame_size,
#         batch_size=batch_size,
#         streaming=streaming,
#         num_workers=2,
#     )


# def create_custom_datamodule(
#     dataset_path: str,
#     batch_size: int = 4,
#     num_frames: int = 16,
#     frame_size: Tuple[int, int] = (224, 224)
# ):
#     """Create DataModule for custom local video dataset"""
    
#     return VideoDeepfakeDataModule(
#         train_dataset=dataset_path,
#         num_frames=num_frames,
#         frame_size=frame_size,
#         batch_size=batch_size,
#         streaming=False,  # Local dataset
#         num_workers=2,
#     )


# Test the data module
if __name__ == "__main__":
    print("Testing MAVOS-DD Dataset...")

    load_dataset("unibuc-cs/MAVOS-DD", streaming=True, split="train", trust_remote_code=True).take(1)
    
#     # Test MAVOS-DD dataset
#     mavos_datamodule = create_mavos_dd_datamodule(
#         batch_size=2,
#         num_frames=8,
#         frame_size=(224, 224),
#         streaming=True,
#         max_samples=10,  # Small sample for testing
#         language_filter="english",  # Filter to English only
#         open_set_mode="indomain"  # Use indomain split
#     )
    
#     # Setup
#     mavos_datamodule.setup()
    
#     # Test dataloaders
#     train_loader = mavos_datamodule.train_dataloader()
    
#     print("Testing MAVOS-DD data loading...")
    
#     # Test one batch
#     for batch_idx, (videos, labels) in enumerate(train_loader):
#         print(f"Batch {batch_idx}:")
#         print(f"  Videos shape: {videos.shape}")  # Should be [batch_size, num_frames, 3, H, W]
#         print(f"  Labels shape: {labels.shape}")  # Should be [batch_size]
#         print(f"  Video range: [{videos.min():.3f}, {videos.max():.3f}]")
#         print(f"  Labels: {labels.tolist()}")
        
#         if batch_idx >= 1:  # Only test a few batches
#             break
    
#     print("MAVOS-DD data loading test completed!")
    
#     # Also test with dummy data for comparison
#     print("\nTesting with dummy data...")
#     datamodule = VideoDeepfakeDataModule(
#         train_dataset=None,  # Will use dummy data
#         batch_size=2,
#         num_frames=8,
#         frame_size=(224, 224),
#         streaming=True,
#         max_train_samples=10,
#         max_val_samples=5,
#     )
    
#     # Setup
#     datamodule.setup()
    
#     # Test dataloaders
#     train_loader = datamodule.train_dataloader()
#     val_loader = datamodule.val_dataloader()
    
#     # Test one batch
#     for batch_idx, (videos, labels) in enumerate(train_loader):
#         print(f"Dummy Batch {batch_idx}:")
#         print(f"  Videos shape: {videos.shape}")  # Should be [batch_size, num_frames, 3, H, W]
#         print(f"  Labels shape: {labels.shape}")  # Should be [batch_size]
#         print(f"  Video range: [{videos.min():.3f}, {videos.max():.3f}]")
        
#         if batch_idx >= 1:  # Only test a few batches
#             break
    
#     print("All data loading tests completed successfully!")

# if __name__ == "__main__":
#     # The test code is included above in the main block
#     # Start small to verify everything works
#     datamodule = create_mavos_dd_datamodule(
#         batch_size=2,           # Small batch
#         num_frames=8,           # Fewer frames  
#         max_samples=100,        # Limit samples
#         streaming=False
#     )

#     # Test loading
#     datamodule.setup()
#     train_loader = datamodule.train_dataloader()

#     for batch in train_loader:
#         videos, labels = batch
#         print(f"Success! Videos: {videos.shape}, Labels: {labels.shape}")
#         break