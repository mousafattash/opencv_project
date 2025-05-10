import cv2
import numpy as np

def apply_histogram_equalization(frame):
    """
    Apply histogram equalization for contrast enhancement to a video frame
    
    Args:
        frame (numpy.ndarray): Input video frame
        
    Returns:
        numpy.ndarray: Frame with histogram equalization applied
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    # Convert back to BGR for display
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def apply_clahe(frame):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    Args:
        frame (numpy.ndarray): Input video frame
        
    Returns:
        numpy.ndarray: Frame with CLAHE applied
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE
    equalized = clahe.apply(gray)
    
    # Convert back to BGR for display
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)