import cv2
import numpy as np

def apply_edge_detection(frame):
    """
    Apply edge detection filter to a video frame
    
    Args:
        frame (numpy.ndarray): Input video frame
        
    Returns:
        numpy.ndarray: Frame with edge detection applied
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    
    # Convert back to BGR for display
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Alternatively, you can overlay edges on original image
    # edges_overlay = frame.copy()
    # edges_overlay[edges > 0] = [0, 255, 255]  # Yellow color for edges
    
    return edges_colored