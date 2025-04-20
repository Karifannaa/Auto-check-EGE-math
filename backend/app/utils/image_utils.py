"""
Image Utilities Module

This module provides utilities for handling and processing images
before sending them to the OpenRouter API.
"""

import base64
import io
from typing import Optional, Tuple, Dict, Any
from PIL import Image


def resize_image_if_needed(image: Image.Image, max_size: int = 4096) -> Image.Image:
    """
    Resize an image if either dimension exceeds the maximum size.
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension size
        
    Returns:
        Resized image if needed, otherwise the original image
    """
    width, height = image.size
    
    # Check if resizing is needed
    if width <= max_size and height <= max_size:
        return image
    
    # Calculate new dimensions while maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Resize the image
    return image.resize((new_width, new_height), Image.LANCZOS)


def enhance_image_quality(image: Image.Image, contrast_factor: float = 1.2) -> Image.Image:
    """
    Enhance image quality by adjusting contrast.
    
    Args:
        image: PIL Image object
        contrast_factor: Factor to adjust contrast (1.0 is original)
        
    Returns:
        Enhanced image
    """
    from PIL import ImageEnhance
    
    # Convert to RGB if in another mode (like RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(contrast_factor)


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Convert a PIL Image to a base64 encoded string.
    
    Args:
        image: PIL Image object
        format: Image format (JPEG, PNG, etc.)
        
    Returns:
        Base64 encoded string of the image
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def prepare_image_for_api(
    image: Image.Image, 
    resize: bool = True,
    enhance: bool = False,
    max_size: int = 4096,
    contrast_factor: float = 1.2
) -> Dict[str, Any]:
    """
    Prepare an image for sending to the OpenRouter API.
    
    Args:
        image: PIL Image object
        resize: Whether to resize the image if needed
        enhance: Whether to enhance the image quality
        max_size: Maximum dimension size for resizing
        contrast_factor: Factor to adjust contrast if enhancing
        
    Returns:
        Dictionary with image data in the format expected by the API
    """
    # Process the image as needed
    if resize:
        image = resize_image_if_needed(image, max_size)
    
    if enhance:
        image = enhance_image_quality(image, contrast_factor)
    
    # Convert to base64
    base64_image = image_to_base64(image)
    
    # Return in the format expected by OpenRouter API
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }


def load_and_prepare_image(
    image_path: str,
    resize: bool = True,
    enhance: bool = False,
    max_size: int = 4096,
    contrast_factor: float = 1.2
) -> Dict[str, Any]:
    """
    Load an image from a file and prepare it for the API.
    
    Args:
        image_path: Path to the image file
        resize: Whether to resize the image if needed
        enhance: Whether to enhance the image quality
        max_size: Maximum dimension size for resizing
        contrast_factor: Factor to adjust contrast if enhancing
        
    Returns:
        Dictionary with image data in the format expected by the API
    """
    # Load the image
    image = Image.open(image_path)
    
    # Prepare and return
    return prepare_image_for_api(
        image, 
        resize=resize, 
        enhance=enhance, 
        max_size=max_size, 
        contrast_factor=contrast_factor
    )
