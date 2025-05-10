import cv2
import numpy as np

def apply_soft_blur(frame, kernel_size=15):
    """
    Apply soft blur filter to a video frame
    
    Args:
        frame (numpy.ndarray): Input video frame
        kernel_size (int): Size of the blur kernel (default: 15)
        
    Returns:
        numpy.ndarray: Frame with soft blur applied
    """
    # Apply Gaussian blur for a soft appearance
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    # For a more aesthetic soft glow effect, we can blend the original with the blurred
    soft_glow = cv2.addWeighted(frame, 0.3, blurred, 0.7, 0)
    
    return soft_glow