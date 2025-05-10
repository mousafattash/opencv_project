import cv2
import numpy as np

def apply_cartoon_filter(frame):
    """
    Apply cartoon filter to a video frame
    
    Args:
        frame (numpy.ndarray): Input video frame
        
    Returns:
        numpy.ndarray: Frame with cartoon effect applied
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur to reduce noise and keep edges sharp
    gray = cv2.medianBlur(gray, 5)
    
    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY, 9, 9)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    # This creates the cartoon-like effect
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    
    # Combine edges with color image
    # Convert edges to 3-channel for bitwise operations
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Bitwise AND operation to overlay edges on the color image
    cartoon = cv2.bitwise_and(color, edges_3channel)
    
    return cartoon