import os
import logging
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        # Try to auto-detect tesseract path
        self.tesseract_cmd = self._find_tesseract()
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
    
    def _find_tesseract(self):
        """Find tesseract executable path"""
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",  # Windows
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",  # Windows x86
            "/usr/bin/tesseract",  # Linux
            "/usr/local/bin/tesseract",  # macOS/Linux
            "/opt/homebrew/bin/tesseract",  # macOS Apple Silicon
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Try which command on Unix-like systems
        try:
            import subprocess
            result = subprocess.run(["which", "tesseract"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        logger.warning("Tesseract not found. Please install Tesseract OCR.")
        return None
    
    def _preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.MedianFilter())
        
        return image
    
    def extract_text_from_image(self, image_file: FileStorage) -> str:
        """Extract text from uploaded image file"""
        try:
            image = Image.open(image_file.stream)
            processed_image = self._preprocess_image(image)
            
            # Use tesseract with optimized config
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:()[]{}<>+-=*/\\|&%$#@_"'
            extracted_text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            return extracted_text.strip()
        
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return ""
    
    def extract_text_from_image_file(self, image_path: str) -> str:
        """Extract text from image file path"""
        try:
            image = Image.open(image_path)
            processed_image = self._preprocess_image(image)
            
            custom_config = r'--oem 3 --psm 6'
            extracted_text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            return extracted_text.strip()
        
        except Exception as e:
            logger.error(f"Error processing image file: {str(e)}")
            return ""