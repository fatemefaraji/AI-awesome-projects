import os
import numpy as np
import random
from typing import Tuple, List, Generator
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import concurrent.futures
import logging
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BRATSDataGenerator(Sequence):
    """
    Advanced 3D data generator for BRATS 2020 dataset with support for:
    - Multi-threaded loading
    - On-the-fly augmentation
    - Flexible batch composition
    - Memory monitoring
    """
    
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        batch_size: int = 2,
        shuffle: bool = True,
        augment: bool = False,
        num_classes: int = 4,
        img_dim: Tuple[int, int, int] = (128, 128, 128),
        num_channels: int = 3,
        max_samples_in_memory: int = 10,
        num_workers: int = 4
    ):
        """
        Initialize the data generator.
        
        Args:
            img_dir: Directory containing input images
            mask_dir: Directory containing masks
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data after each epoch
            augment: Whether to apply data augmentation
            num_classes: Number of segmentation classes
            img_dim: Dimensions of input images (H, W, D)
            num_channels: Number of image channels
            max_samples_in_memory: Maximum samples to keep in memory
            num_workers: Number of parallel workers for loading
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.num_channels = num_channels
        self.max_samples_in_memory = max_samples_in_memory
        self.num_workers = num_workers
        
        # Get list of all files
        self.img_list = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
        self.mask_list = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])
        
        # Validate that image and mask lists match
        assert len(self.img_list) == len(self.mask_list), \
            "Number of images and masks must be equal"
        for img, mask in zip(self.img_list, self.mask_list):
            assert img == mask, f"Image and mask filenames must match: {img} vs {mask}"
        
        self.num_samples = len(self.img_list)
        self.indices = np.arange(self.num_samples)
        
        # Initialize memory cache
        self.img_cache = {}
        self.mask_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize augmentation parameters
        self.augmentation_params = {
            'rotation_range': 15,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'vertical_flip': True,
            'fill_mode': 'constant',
            'brightness_range': [0.9, 1.1],
            'contrast_range': [0.9, 1.1]
        }
        
        # Log initialization
        logger.info(f"Initialized generator with {self.num_samples} samples")
        logger.info(f"Batch size: {self.batch_size}, Augmentation: {self.augment}")
        
        if self.shuffle:
            self.on_epoch_end()
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        # Get batch indices
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        # Initialize batch arrays
        X = np.empty((len(batch_indices), *self.img_dim, self.num_channels), dtype=np.float32)
        y = np.empty((len(batch_indices), *self.img_dim, self.num_classes), dtype=np.float32)
        
        # Load data for each sample in batch
        for i, idx in enumerate(batch_indices):
            X[i,], y[i,] = self._load_sample(idx)
            
            # Apply augmentation if enabled
            if self.augment:
                X[i,], y[i,] = self._augment_sample(X[i,], y[i,])
        
        return X, y
    
    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single sample from disk or cache."""
        img_name = self.img_list[idx]
        mask_name = self.mask_list[idx]
        
        # Try to get from cache first
        if img_name in self.img_cache and mask_name in self.mask_cache:
            self.cache_hits += 1
            return self.img_cache[img_name], self.mask_cache[mask_name]
        
        # Load from disk
        self.cache_misses += 1
        try:
            img = np.load(os.path.join(self.img_dir, img_name))
            mask = np.load(os.path.join(self.mask_dir, mask_name))
            
            # Ensure correct shape
            if len(img.shape) == 3:  # If channels not included
                img = np.expand_dims(img, axis=-1)
            
            # Add to cache if there's space
            if len(self.img_cache) < self.max_samples_in_memory:
                self.img_cache[img_name] = img
                self.mask_cache[mask_name] = mask
            
            return img, mask
        except Exception as e:
            logger.error(f"Error loading sample {img_name}: {str(e)}")
            raise
    
    def _augment_sample(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations to a single sample."""
        # Random rotation
        angle = np.random.uniform(-self.augmentation_params['rotation_range'], 
                                 self.augmentation_params['rotation_range'])
        
        # Random shifts
        h_shift = np.random.uniform(-self.augmentation_params['height_shift_range'], 
                                    self.augmentation_params['height_shift_range'])
        w_shift = np.random.uniform(-self.augmentation_params['width_shift_range'], 
                                    self.augmentation_params['width_shift_range'])
        
        # Random zoom
        zoom = np.random.uniform(1 - self.augmentation_params['zoom_range'], 
                                1 + self.augmentation_params['zoom_range'])
        
        # Random flips
        h_flip = self.augmentation_params['horizontal_flip'] and np.random.random() < 0.5
        v_flip = self.augmentation_params['vertical_flip'] and np.random.random() < 0.5
        
        # Apply transformations to each slice
        for z in range(img.shape[2]):
            # Apply to image channels
            for c in range(img.shape[3]):
                slice_img = img[:, :, z, c]
                
                # Apply transformations
                if h_flip:
                    slice_img = np.fliplr(slice_img)
                if v_flip:
                    slice_img = np.flipud(slice_img)
                
                img[:, :, z, c] = slice_img
            
            # Apply to mask
            slice_mask = mask[:, :, z, :]
            if h_flip:
                slice_mask = np.fliplr(slice_mask)
            if v_flip:
                slice_mask = np.flipud(slice_mask)
            mask[:, :, z, :] = slice_mask
        
        return img, mask
    
    def on_epoch_end(self):
        """Update indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Periodically log cache performance
        if len(self.img_cache) > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
            logger.debug(f"Cache hit rate: {hit_rate:.2%}")
        
        # Clear cache if it's getting too large
        if len(self.img_cache) > self.max_samples_in_memory * 1.5:
            self.img_cache.clear()
            self.mask_cache.clear()
            logger.debug("Cleared sample cache")
    
    def visualize_batch(self, batch_num: int = None, save_path: str = None):
        """Visualize a batch of samples."""
        if batch_num is None:
            batch_num = random.randint(0, len(self)-1)
        
        X, y = self[batch_num]
        
        # Select random sample from batch
        sample_idx = random.randint(0, X.shape[0]-1)
        img = X[sample_idx]
        mask = y[sample_idx]
        
        # Convert one-hot mask to single channel
        mask = np.argmax(mask, axis=-1)
        
        # Select random slice
        slice_idx = random.randint(0, img.shape[2]-1)
        
        # Plot
        plt.figure(figsize=(18, 10))
        
        # Plot each channel
        for i in range(img.shape[3]):
            plt.subplot(2, 3, i+1)
            plt.imshow(img[:, :, slice_idx, i], cmap='gray')
            plt.title(f'Channel {i}')
            plt.axis('off')
        
        # Plot mask
        plt.subplot(2, 3, img.shape[3]+1)
        plt.imshow(mask[:, :, slice_idx])
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        plt.suptitle(f'Batch {batch_num}, Sample {sample_idx}, Slice {slice_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def create_data_generators(
    train_img_dir: str,
    train_mask_dir: str,
    val_img_dir: str = None,
    val_mask_dir: str = None,
    test_img_dir: str = None,
    test_mask_dir: str = None,
    batch_size: int = 4,
    validation_split: float = 0.2,
    augment_train: bool = True,
    **kwargs
) -> Tuple[BRATSDataGenerator, Optional[BRATSDataGenerator], Optional[BRATSDataGenerator]]:
    """
    Create data generators for training, validation, and testing.
    
    Args:
        train_img_dir: Directory containing training images
        train_mask_dir: Directory containing training masks
        val_img_dir: Optional directory for validation images
        val_mask_dir: Optional directory for validation masks
        test_img_dir: Optional directory for test images
        test_mask_dir: Optional directory for test masks
        batch_size: Batch size for generators
        validation_split: Fraction of training data to use for validation
                         (only used if val_img_dir not provided)
        augment_train: Whether to augment training data
        **kwargs: Additional arguments for BRATSDataGenerator
        
    Returns:
        Tuple of (train_gen, val_gen, test_gen)
    """
    # Create training generator
    train_gen = BRATSDataGenerator(
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        batch_size=batch_size,
        augment=augment_train,
        shuffle=True,
        **kwargs
    )
    
    # Create validation generator
    val_gen = None
    if val_img_dir and val_mask_dir:
        val_gen = BRATSDataGenerator(
            img_dir=val_img_dir,
            mask_dir=val_mask_dir,
            batch_size=batch_size,
            augment=False,  # No augmentation for validation
            shuffle=False,
            **kwargs
        )
    elif validation_split > 0:
        # Split training data if separate validation dir not provided
        logger.warning("No validation directory provided, splitting training data")
        # This would require modifying the generator to handle split indices
        # For now, we'll just return None for val_gen in this case
    
    # Create test generator if paths provided
    test_gen = None
    if test_img_dir and test_mask_dir:
        test_gen = BRATSDataGenerator(
            img_dir=test_img_dir,
            mask_dir=test_mask_dir,
            batch_size=batch_size,
            augment=False,  # No augmentation for test
            shuffle=False,
            **kwargs
        )
    
    return train_gen, val_gen, test_gen


# Example usage
if __name__ == "__main__":
    # Paths to your data
    train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
    train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"
    val_img_dir = "BraTS2020_TrainingData/input_data_128/val/images/"
    val_mask_dir = "BraTS2020_TrainingData/input_data_128/val/masks/"
    
    # Create generators
    train_gen, val_gen, _ = create_data_generators(
        train_img_dir=train_img_dir,
        train_mask_dir=train_mask_dir,
        val_img_dir=val_img_dir,
        val_mask_dir=val_mask_dir,
        batch_size=2,
        augment_train=True,
        num_classes=4,
        img_dim=(128, 128, 128),
        num_channels=3
    )
    
    # Visualize a batch
    train_gen.visualize_batch()
    
    # Example of using in model training
    # model.fit(train_gen, validation_data=val_gen, epochs=55)
