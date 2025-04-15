"""
Advanced MRI Image Denoising Pipeline

This script provides a comprehensive denoising framework for MRI images, implementing:
- Gaussian, Bilateral, Total Variation (TV), Wavelet, Shift-Invariant Wavelet
- Anisotropic Diffusion, Non-Local Means (NLM via skimage and OpenCV)
- Block-Matching and 3D Filtering (BM3D)
- Markov Random Field (MRF) with Iterated Conditional Modes (ICM)

The code is optimized for performance, modularity, and clarity, with PSNR metrics
to evaluate denoising quality. It assumes input images are in TIFF format and
processes them in grayscale.

Author: Adapted and enhanced from Sreenivas Bhattiprolu's code
License: Free to use with acknowledgment
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, img_as_float, img_as_ubyte
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, cycle_spin, denoise_nl_means,
                                 estimate_sigma)
from skimage.metrics import peak_signal_noise_ratio
from scipy import ndimage as nd
from medpy.filter.smoothing import anisotropic_diffusion
import bm3d
import pydicom
import logging

# Configure logging for better debugging and tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
BASE_PATH = "images/MRI_images"
NOISY_IMAGE = os.path.join(BASE_PATH, "MRI_noisy.tif")
REF_IMAGE = os.path.join(BASE_PATH, "MRI_clean.tif")
OUTPUT_DIR = BASE_PATH
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dicom_to_tiff(dicom_path, output_tiff):
    """Convert DICOM to TIFF, extracting pixel data."""
    try:
        dataset = pydicom.dcmread(dicom_path)
        img = dataset.pixel_array
        plt.imsave(output_tiff, img, cmap='gray')
        logger.info(f"Converted DICOM to TIFF: {output_tiff}")
        return img
    except Exception as e:
        logger.error(f"Error converting DICOM: {e}")
        raise

def compute_psnr(ref_img, test_img):
    """Calculate PSNR between reference and test images."""
    return peak_signal_noise_ratio(ref_img, test_img)

def save_image(img, filename, cmap='gray'):
    """Save image to file with error handling."""
    try:
        plt.imsave(os.path.join(OUTPUT_DIR, filename), img, cmap=cmap)
        logger.info(f"Saved image: {filename}")
    except Exception as e:
        logger.error(f"Error saving image {filename}: {e}")

# Denoising Functions
def gaussian_denoise(noisy_img, ref_img, sigma=5):
    """Apply Gaussian denoising."""
    start_time = time.time()
    gaussian_img = nd.gaussian_filter(noisy_img, sigma=sigma)
    psnr = compute_psnr(ref_img, gaussian_img)
    logger.info(f"Gaussian Denoising - PSNR: {psnr:.4f}, Time: {time.time() - start_time:.2f}s")
    save_image(gaussian_img, "Gaussian_smoothed.tif")
    return gaussian_img, psnr

def bilateral_denoise(noisy_img, ref_img, sigma_spatial=15):
    """Apply Bilateral denoising."""
    start_time = time.time()
    bilateral_img = denoise_bilateral(noisy_img, sigma_spatial=sigma_spatial, multichannel=False)
    psnr = compute_psnr(ref_img, bilateral_img)
    logger.info(f"Bilateral Denoising - PSNR: {psnr:.4f}, Time: {time.time() - start_time:.2f}s")
    save_image(bilateral_img, "bilateral_smoothed.tif")
    return bilateral_img, psnr

def tv_denoise(noisy_img, ref_img, weight=0.3):
    """Apply Total Variation denoising."""
    start_time = time.time()
    tv_img = denoise_tv_chambolle(noisy_img, weight=weight, multichannel=False)
    psnr = compute_psnr(ref_img, tv_img)
    logger.info(f"TV Denoising - PSNR: {psnr:.4f}, Time: {time.time() - start_time:.2f}s")
    save_image(tv_img, "TV_smoothed.tif")
    return tv_img, psnr

def wavelet_denoise(noisy_img, ref_img):
    """Apply Wavelet denoising."""
    start_time = time.time()
    wavelet_img = denoise_wavelet(noisy_img, multichannel=False,
                                  method='BayesShrink', mode='soft',
                                  rescale_sigma=True)
    psnr = compute_psnr(ref_img, wavelet_img)
    logger.info(f"Wavelet Denoising - PSNR: {psnr:.4f}, Time: {time.time() - start_time:.2f}s")
    save_image(wavelet_img, "wavelet_smoothed.tif")
    return wavelet_img, psnr

def shift_invariant_wavelet_denoise(noisy_img, ref_img, max_shifts=3):
    """Apply Shift-Invariant Wavelet denoising using cycle spinning."""
    start_time = time.time()
    denoise_kwargs = dict(multichannel=False, wavelet='db1', method='BayesShrink',
                          rescale_sigma=True)
    shift_inv_img = cycle_spin(noisy_img, func=denoise_wavelet, max_shifts=max_shifts,
                               func_kw=denoise_kwargs, multichannel=False)
    psnr = compute_psnr(ref_img, shift_inv_img)
    logger.info(f"Shift-Invariant Wavelet Denoising - PSNR: {psnr:.4f}, Time: {time.time() - start_time:.2f}s")
    save_image(shift_inv_img, "Shift_Inv_wavelet_smoothed.tif")
    return shift_inv_img, psnr

def anisotropic_denoise(noisy_img, ref_img, niter=50, kappa=50, gamma=0.2):
    """Apply Anisotropic Diffusion denoising."""
    start_time = time.time()
    anisotropic_img = anisotropic_diffusion(noisy_img, niter=niter, kappa=kappa,
                                           gamma=gamma, option=2)
    psnr = compute_psnr(ref_img, anisotropic_img)
    logger.info(f"Anisotropic Denoising - PSNR: {psnr:.4f}, Time: {time.time() - start_time:.2f}s")
    save_image(anisotropic_img, "anisotropic_denoised.tif")
    return anisotropic_img, psnr

def nlm_skimage_denoise(noisy_img, ref_img):
    """Apply Non-Local Means denoising using skimage."""
    start_time = time.time()
    sigma_est = np.mean(estimate_sigma(noisy_img, multichannel=False))
    nlm_img = denoise_nl_means(noisy_img, h=1.15 * sigma_est, fast_mode=True,
                               patch_size=9, patch_distance=5, multichannel=False)
    psnr = compute_psnr(ref_img, nlm_img)
    logger.info(f"NLM (skimage) Denoising - PSNR: {psnr:.4f}, Time: {time.time() - start_time:.2f}s")
    save_image(img_as_ubyte(nlm_img), "NLM_skimage_denoised.tif")
    return nlm_img, psnr

def nlm_opencv_denoise(noisy_img, ref_img):
    """Apply Non-Local Means denoising using OpenCV."""
    start_time = time.time()
    # Convert to 8-bit for OpenCV
    noisy_8bit = img_as_ubyte(noisy_img)
    nlm_img = cv2.fastNlMeansDenoising(noisy_8bit, None, h=3, templateWindowSize=7, searchWindowSize=21)
    nlm_img_float = img_as_float(nlm_img)
    psnr = compute_psnr(ref_img, nlm_img_float)
    logger.info(f"NLM (OpenCV) Denoising - PSNR: {psnr:.4f}, Time: {time.time() - start_time:.2f}s")
    save_image(nlm_img_float, "NLM_CV2_denoised.tif")
    return nlm_img_float, psnr

def bm3d_denoise(noisy_img, ref_img, sigma_psd=0.2):
    """Apply BM3D denoising."""
    start_time = time.time()
    bm3d_img = bm3d.bm3d(noisy_img, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    psnr = compute_psnr(ref_img, bm3d_img)
    logger.info(f"BM3D Denoising - PSNR: {psnr:.4f}, Time: {time.time() - start_time:.2f}s")
    save_image(bm3d_img, "BM3D_denoised.tif")
    return bm3d_img, psnr

def mrf_icm_denoise(noisy_img, ref_img, iterations=5, beta=1.0, sigma2=5.0):
    """
    Apply MRF-based denoising using Iterated Conditional Modes (ICM).
    Optimized for speed and quality with robust potentials.
    """
    start_time = time.time()
    img = noisy_img.copy() * 255  # Scale to [0, 255]
    height, width = img.shape
    denoised_img = img.copy()

    def pot(fi, fj, sigma=sigma2):
        """Lorentzian potential for robust smoothness (non-convex)."""
        z = (fi - fj) / sigma
        return np.log(1 + 0.5 * z**2)

    for iter in range(iterations):
        logger.info(f"MRF ICM Iteration {iter + 1}/{iterations}")
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                min_energy = float('inf')
                best_x = denoised_img[i, j]
                # Test a subset of gray levels around current value for efficiency
                current_val = int(denoised_img[i, j])
                gray_range = range(max(0, current_val - 50), min(256, current_val + 51))
                
                for x in gray_range:
                    # Data term: Quadratic potential
                    data_term = ((img[i, j] - x) ** 2) / (2.0 * sigma2)
                    # Smoothness term: Lorentzian potential for neighbors
                    smooth_term = beta * (
                        pot(denoised_img[i-1, j], x) +
                        pot(denoised_img[i+1, j], x) +
                        pot(denoised_img[i, j-1], x) +
                        pot(denoised_img[i, j+1], x)
                    )
                    energy = data_term + smooth_term
                    if energy < min_energy:
                        min_energy = energy
                        best_x = x
                denoised_img[i, j] = best_x
        
        # Save intermediate result
        save_image(denoised_img / 255.0, f"MRF_ICM_iter_{iter+1}.tif")
    
    denoised_img = denoised_img / 255.0  # Rescale back to [0, 1]
    psnr = compute_psnr(ref_img, denoised_img)
    logger.info(f"MRF ICM Denoising - PSNR: {psnr:.4f}, Time: {time.time() - start_time:.2f}s")
    save_image(denoised_img, "MRF_ICM_denoised.tif")
    return denoised_img, psnr

def plot_results(images, psnrs, titles):
    """Plot all denoised images with PSNR values."""
    n = len(images)
    plt.figure(figsize=(15, 5 * n // 3))
    for i, (img, title, psnr) in enumerate(zip(images, titles, psnrs)):
        plt.subplot(n // 3 + 1, 3, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{title}\nPSNR: {psnr:.4f}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "denoising_comparison.png"))
    plt.show()

def main():
    """Main function to run the denoising pipeline."""
    logger.info("Starting MRI Denoising Pipeline")
    
    # Load images
    noisy_img = img_as_float(io.imread(NOISY_IMAGE, as_gray=True))
    ref_img = img_as_float(io.imread(REF_IMAGE, as_gray=True))
    
    # Compute noisy image PSNR
    noise_psnr = compute_psnr(ref_img, noisy_img)
    logger.info(f"Noisy Image PSNR: {noise_psnr:.4f}")
    
    # Run denoising methods
    methods = [
        (gaussian_denoise, {"sigma": 5}),
        (bilateral_denoise, {"sigma_spatial": 15}),
        (tv_denoise, {"weight": 0.3}),
        (wavelet_denoise, {}),
        (shift_invariant_wavelet_denoise, {"max_shifts": 3}),
        (anisotropic_denoise, {"niter": 50, "kappa": 50, "gamma": 0.2}),
        (nlm_skimage_denoise, {}),
        (nlm_opencv_denoise, {}),
        (bm3d_denoise, {"sigma_psd": 0.2}),
        (mrf_icm_denoise, {"iterations": 5, "beta": 1.0, "sigma2": 5.0}),
    ]
    
    images = [noisy_img]
    psnrs = [noise_psnr]
    titles = ["Noisy Image"]
    
    for method, kwargs in methods:
        img, psnr = method(noisy_img, ref_img, **kwargs)
        images.append(img)
        psnrs.append(psnr)
        titles.append(method.__name__.replace("_denoise", "").replace("_", " ").title())
    
    # Plot results
    plot_results(images, psnrs, titles)
    logger.info("Denoising Pipeline Completed")

if __name__ == "__main__":
    main()