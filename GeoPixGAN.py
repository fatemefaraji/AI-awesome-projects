#!/usr/bin/env python
"""
GeoPixGAN: Generating Realistic Geological Images with Pix2Pix GAN

This project uses a Pix2Pix GAN to transform segmented sandstone images into realistic
grayscale images resembling geological thin sections. It includes robust data loading,
preprocessing, model training, and evaluation with SSIM and PSNR metrics. Visualizations
compare input masks, generated images, and ground truth.

Author: Adapted and enhanced from Sreenivas Bhattiprolu's code
License: Free to use with acknowledgment
Video Reference: https://youtu.be/my7LEgYTJto
Dataset: https://drive.google.com/file/d/1HWtBaSa-LTyAMgf2uaz1T9o1sTWDBajU/view

Dependencies: tensorflow, keras, opencv-python, numpy, matplotlib, scikit-image
"""

import os
import glob
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from tensorflow.keras.models import load_model
from numpy.random import randint
from numpy import vstack
from pix2pix_model import define_discriminator, define_generator, define_gan, train

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SIZE_X = 256
SIZE_Y = 256
DATA_DIR = "sandstone"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "sandstone_gan.h5")
TEST_MASK_PATH = os.path.join(DATA_DIR, "test_mask.tif")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_images(image_dir, mask_dir, size=(SIZE_Y, SIZE_X)):
    """Load and preprocess source (masks) and target (images)."""
    try:
        tar_images = []
        src_images = []
        
        # Load target images
        for img_path in glob.glob(os.path.join(image_dir, "*.tif")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue
            img = cv2.resize(img, size)
            tar_images.append(img)
        
        # Load source masks
        for mask_path in glob.glob(os.path.join(mask_dir, "*.tif")):
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            if mask is None:
                logger.warning(f"Failed to load mask: {mask_path}")
                continue
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            src_images.append(mask)
        
        tar_images = np.array(tar_images, dtype=np.float32)
        src_images = np.array(src_images, dtype=np.float32)
        
        logger.info(f"Loaded {len(tar_images)} target images and {len(src_images)} source masks")
        return src_images, tar_images
    except Exception as e:
        logger.error(f"Error loading images: {e}")
        raise

def preprocess_data(src_images, tar_images):
    """Scale images from [0, 255] to [-1, 1] for GAN training."""
    X1 = (src_images - 127.5) / 127.5
    X2 = (tar_images - 127.5) / 127.5
    return [X1, X2]

def visualize_samples(src_images, tar_images, n_samples=3):
    """Visualize sample source and target images."""
    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        # Source (mask)
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(src_images[i, :, :, 0], cmap='gray')
        plt.title(f"Mask {i+1}")
        plt.axis('off')
        # Target (image)
        plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(tar_images[i, :, :, 0], cmap='gray')
        plt.title(f"Image {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"))
    plt.show()

def define_models(image_shape):
    """Define Pix2Pix discriminator, generator, and GAN models."""
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
    gan_model = define_gan(g_model, d_model, image_shape)
    logger.info("Models defined successfully")
    return d_model, g_model, gan_model

def compute_metrics(real_img, gen_img):
    """Compute SSIM and PSNR between real and generated images."""
    real_img = (real_img + 1) / 2.0  # Scale back to [0, 1]
    gen_img = (gen_img + 1) / 2.0
    ssim_val = ssim(real_img[:, :, 0], gen_img[:, :, 0], data_range=1.0)
    psnr_val = psnr(real_img[:, :, 0], gen_img[:, :, 0], data_range=1.0)
    return ssim_val, psnr_val

def plot_images(src_img, gen_img, tar_img, ssim_val=None, psnr_val=None):
    """Plot source, generated, and target images with metrics."""
    images = vstack((src_img, gen_img, tar_img))
    images = (images + 1) / 2.0  # Scale to [0, 1]
    titles = ['Input Mask', 'Generated Image', 'Ground Truth']
    
    plt.figure(figsize=(12, 4))
    for i in range(len(images)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i, :, :, 0], cmap='gray')
        title = titles[i]
        if i == 1 and ssim_val is not None and psnr_val is not None:
            title += f'\nSSIM: {ssim_val:.4f}, PSNR: {psnr_val:.4f}'
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison.png"))
    plt.show()

def test_model(model, dataset, n_samples=3):
    """Test the trained model on random dataset samples."""
    X1, X2 = dataset
    indices = randint(0, len(X1), n_samples)
    for ix in indices:
        src_img, tar_img = X1[ix:ix+1], X2[ix:ix+1]
        gen_img = model.predict(src_img)
        ssim_val, psnr_val = compute_metrics(tar_img[0], gen_img[0])
        logger.info(f"Sample {ix}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.4f}")
        plot_images(src_img, gen_img, tar_img, ssim_val, psnr_val)

def test_single_image(model, test_mask_path, size=(SIZE_Y, SIZE_X)):
    """Test the model on a single test mask."""
    try:
        test_img = cv2.imread(test_mask_path, cv2.IMREAD_COLOR)
        if test_img is None:
            raise ValueError(f"Failed to load test mask: {test_mask_path}")
        test_img = cv2.resize(test_img, size, interpolation=cv2.INTER_NEAREST)
        test_img = (test_img - 127.5) / 127.5
        test_img = np.expand_dims(test_img, axis=0)
        
        gen_img = model.predict(test_img)
        gen_img = (gen_img + 1) / 2.0  # Scale to [0, 1]
        
        plt.figure(figsize=(6, 6))
        plt.imshow(gen_img[0, :, :, 0], cmap='gray')
        plt.title("Generated Test Image")
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, "test_generated.png"))
        plt.show()
    except Exception as e:
        logger.error(f"Error testing single image: {e}")

def main():
    """Main function to run the GeoPixGAN pipeline."""
    logger.info("Starting GeoPixGAN Pipeline")
    
    # Load and preprocess data
    src_images, tar_images = load_images(
        os.path.join(DATA_DIR, "images"),
        os.path.join(DATA_DIR, "masks")
    )
    
    # Visualize samples
    visualize_samples(src_images, tar_images)
    
    # Preprocess data
    dataset = preprocess_data(src_images, tar_images)
    image_shape = src_images.shape[1:]
    
    # Define models
    d_model, g_model, gan_model = define_models(image_shape)
    
    # Train models
    start_time = datetime.now()
    train(d_model, g_model, gan_model, dataset, n_epochs=50, n_batch=1)
    execution_time = datetime.now() - start_time
    logger.info(f"Training completed. Execution time: {execution_time}")
    
    # Save generator model
    g_model.save(MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    
    # Test model on random samples
    logger.info("Testing model on random samples")
    test_model(g_model, dataset)
    
    # Test model on single test mask
    if os.path.exists(TEST_MASK_PATH):
        logger.info("Testing model on single test mask")
        test_single_image(g_model, TEST_MASK_PATH)
    
    logger.info("GeoPixGAN Pipeline Completed")

if __name__ == "__main__":
    main()