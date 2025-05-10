import cv2
import numpy as np

def apply_grayscale_quantization(frame, levels=8):
    """
    Apply grayscale quantization filter to a video frame
    
    Args:
        frame (numpy.ndarray): Input video frame
        levels (int): Number of gray levels (default: 8)
        
    Returns:
        numpy.ndarray: Frame with grayscale quantization applied
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply quantization
    # Formula: quantized = (gray // (256 // levels)) * (256 // levels)
    quantization_factor = 256 // levels
    quantized = (gray // quantization_factor) * quantization_factor
    
    # Convert back to BGR for display
    return cv2.cvtColor(quantized, cv2.COLOR_GRAY2BGR)