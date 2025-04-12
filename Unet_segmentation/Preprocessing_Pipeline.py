import os
import numpy as np
import nibabel as nib
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tifffile import imsave
from tensorflow.keras.utils import to_categorical
from concurrent.futures import ProcessPoolExecutor
import logging
from typing import Tuple, List, Optional
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BRATSPreprocessor:
    """
    Advanced preprocessing pipeline for BRATS 2020 dataset.
    Handles loading, normalization, cropping, and saving of 3D MRI volumes and masks.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.scaler = MinMaxScaler()
        self.valid_modalities = ['flair', 't1', 't1ce', 't2']
        
        # Create output directories if they don't exist
        os.makedirs(config['output_img_dir'], exist_ok=True)
        os.makedirs(config['output_mask_dir'], exist_ok=True)
        
    def load_nifti_file(self, filepath: str) -> np.ndarray:
        """
        Load a NIfTI file and return its data as numpy array.
        
        Args:
            filepath: Path to the NIfTI file
            
        Returns:
            Numpy array containing the image data
        """
        try:
            img = nib.load(filepath)
            return img.get_fdata()
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {str(e)}")
            raise
            
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize a 3D volume using MinMax scaling.
        
        Args:
            volume: 3D numpy array to normalize
            
        Returns:
            Normalized 3D numpy array
        """
        original_shape = volume.shape
        volume = self.scaler.fit_transform(volume.reshape(-1, volume.shape[-1]))
        return volume.reshape(original_shape)
    
    def process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Process the segmentation mask:
        - Convert to uint8
        - Reassign label 4 to 3 (for consistency)
        - Optionally one-hot encode
        
        Args:
            mask: Input segmentation mask
            
        Returns:
            Processed mask
        """
        mask = mask.astype(np.uint8)
        mask[mask == 4] = 3  # Reassign label 4 to 3
        
        if self.config['one_hot_encode']:
            mask = to_categorical(mask, num_classes=self.config['num_classes'])
            
        return mask
    
    def crop_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Crop the volume to the specified dimensions.
        
        Args:
            volume: Input volume to crop
            
        Returns:
            Cropped volume
        """
        x_start, x_end = self.config['crop_x']
        y_start, y_end = self.config['crop_y']
        z_start, z_end = self.config['crop_z']
        
        return volume[x_start:x_end, y_start:y_end, z_start:z_end]
    
    def has_sufficient_annotations(self, mask: np.ndarray) -> bool:
        """
        Check if the mask contains sufficient non-zero annotations.
        
        Args:
            mask: Segmentation mask to check
            
        Returns:
            True if mask has sufficient annotations, False otherwise
        """
        if self.config['one_hot_encode']:
            # For one-hot encoded masks, check class 1, 2, and 3
            class_pixels = np.sum(mask[..., 1:], axis=(0, 1, 2))
            return np.any(class_pixels > self.config['min_annotation_pixels'])
        else:
            # For regular masks
            unique, counts = np.unique(mask, return_counts=True)
            if len(unique) > 1:
                non_zero_ratio = 1 - (counts[0] / counts.sum())
                return non_zero_ratio > self.config['min_annotation_ratio']
            return False
    
    def visualize_sample(
        self, 
        images: List[np.ndarray], 
        mask: np.ndarray, 
        slice_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize a sample slice from the volumes and mask.
        
        Args:
            images: List of modality images
            mask: Corresponding segmentation mask
            slice_idx: Index of slice to visualize (random if None)
            save_path: Path to save the visualization (optional)
        """
        if slice_idx is None:
            slice_idx = np.random.randint(0, mask.shape[2])
            
        modalities = ['FLAIR', 'T1', 'T1CE', 'T2'][:len(images)]
        
        plt.figure(figsize=(15, 10))
        
        # Plot each modality
        for i, (modality, img) in enumerate(zip(modalities, images)):
            plt.subplot(2, 3, i+1)
            plt.imshow(img[:, :, slice_idx], cmap='gray')
            plt.title(modality)
            plt.axis('off')
        
        # Plot the mask
        plt.subplot(2, 3, len(images)+1)
        if self.config['one_hot_encode']:
            # Convert one-hot back to single channel for visualization
            plt.imshow(np.argmax(mask[:, :, slice_idx], axis=-1))
        else:
            plt.imshow(mask[:, :, slice_idx])
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def process_single_case(
        self, 
        case_id: str, 
        modality_paths: dict, 
        mask_path: str
    ) -> Tuple[bool, str]:
        """
        Process a single case (all modalities and mask).
        
        Args:
            case_id: Identifier for the case
            modality_paths: Dictionary mapping modality names to file paths
            mask_path: Path to the segmentation mask
            
        Returns:
            Tuple of (success_flag, message)
        """
        try:
            # Load and normalize each modality
            images = []
            for mod in self.config['selected_modalities']:
                img = self.load_nifti_file(modality_paths[mod])
                img = self.normalize_volume(img)
                images.append(img)
            
            # Load and process mask
            mask = self.load_nifti_file(mask_path)
            mask = self.process_mask(mask)
            
            # Crop volumes
            images = [self.crop_volume(img) for img in images]
            mask = self.crop_volume(mask)
            
            # Check if mask has sufficient annotations
            if not self.has_sufficient_annotations(mask):
                return False, f"Case {case_id} has insufficient annotations"
            
            # Combine modalities into multi-channel image
            combined_img = np.stack(images, axis=-1)
            
            # Save results
            np.save(os.path.join(self.config['output_img_dir'], f'image_{case_id}.npy'), combined_img)
            np.save(os.path.join(self.config['output_mask_dir'], f'mask_{case_id}.npy'), mask)
            
            # Optionally save visualization
            if self.config['save_visualizations']:
                vis_path = os.path.join(self.config['output_vis_dir'], f'case_{case_id}.png')
                self.visualize_sample(images, mask, save_path=vis_path)
            
            return True, f"Successfully processed case {case_id}"
            
        except Exception as e:
            return False, f"Error processing case {case_id}: {str(e)}"
    
    def run(self):
        """
        Run the preprocessing pipeline on all cases in the dataset.
        """
        logger.info("Starting BRATS 2020 preprocessing pipeline")
        
        # Get list of all cases
        case_dirs = sorted(glob.glob(os.path.join(self.config['dataset_path'], '*')))
        logger.info(f"Found {len(case_dirs)} cases in dataset")
        
        # Process each case
        success_count = 0
        for case_dir in case_dirs:
            case_id = os.path.basename(case_dir)
            
            # Get paths for all modalities and mask
            modality_paths = {
                mod: os.path.join(case_dir, f'BraTS20_Training_{case_id}_{mod}.nii')
                for mod in self.valid_modalities
            }
            mask_path = os.path.join(case_dir, f'BraTS20_Training_{case_id}_seg.nii')
            
            # Process the case
            success, message = self.process_single_case(case_id, modality_paths, mask_path)
            if success:
                success_count += 1
                logger.debug(message)
            else:
                logger.warning(message)
        
        logger.info(f"Processing complete. Successfully processed {success_count}/{len(case_dirs)} cases")

def get_default_config():
    """
    Return default configuration for the preprocessor.
    """
    return {
        'dataset_path': 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData',
        'selected_modalities': ['flair', 't1ce', 't2'],  # Omitting t1 as it's often less informative
        'output_img_dir': 'BraTS2020_TrainingData/preprocessed/images',
        'output_mask_dir': 'BraTS2020_TrainingData/preprocessed/masks',
        'output_vis_dir': 'BraTS2020_TrainingData/preprocessed/visualizations',
        'one_hot_encode': True,
        'num_classes': 4,
        'crop_x': (56, 184),  # Crop to 128x128x128 volumes
        'crop_y': (56, 184),
        'crop_z': (13, 141),
        'min_annotation_ratio': 0.01,  # At least 1% non-background voxels
        'min_annotation_pixels': 1000,  # Minimum number of pixels for any non-zero class
        'save_visualizations': True,
        'num_workers': 4  # For parallel processing
    }

def main():
    parser = argparse.ArgumentParser(description='BRATS 2020 Dataset Preprocessing')
    parser.add_argument('--input', type=str, help='Path to input dataset directory')
    parser.add_argument('--output', type=str, help='Base path for output directories')
    parser.add_argument('--visualize', action='store_true', help='Save sample visualizations')
    args = parser.parse_args()
    
    # Get configuration
    config = get_default_config()
    
    # Override with command line arguments if provided
    if args.input:
        config['dataset_path'] = args.input
    if args.output:
        config['output_img_dir'] = os.path.join(args.output, 'images')
        config['output_mask_dir'] = os.path.join(args.output, 'masks')
        config['output_vis_dir'] = os.path.join(args.output, 'visualizations')
    config['save_visualizations'] = args.visualize
    
    # Create and run preprocessor
    preprocessor = BRATSPreprocessor(config)
    preprocessor.run()

if __name__ == "__main__":
    main()
