import cv2
import numpy as np
import os
import sys

# Add the project root to the path to import from shared modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import filter modes
from modes.edge_detection import apply_edge_detection
from modes.grayscale_quantization import apply_grayscale_quantization
from modes.contrast_enhancement import apply_histogram_equalization
from modes.soft_blur import apply_soft_blur
from modes.cartoon_filter import apply_cartoon_filter

def display_menu():
    """
    Display the menu of available filter modes
    """
    print("\n===== Video Filter Modes =====")
    print("1. Edge Detection")
    print("2. Grayscale Quantization")
    print("3. Histogram Equalization (Contrast)")
    print("4. Blurring Filter (Soft Appearance)")
    print("5. Cartoon Filter")
    print("0. Exit")
    print("\nPress the corresponding number key to switch modes")
    print("Press 'q' to quit")

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set initial mode
    current_mode = 0  # 0 means no filter (original)
    display_menu()
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Apply selected filter
        if current_mode == 1:
            processed_frame = apply_edge_detection(frame)
            mode_name = "Edge Detection"
        elif current_mode == 2:
            processed_frame = apply_grayscale_quantization(frame)
            mode_name = "Grayscale Quantization"
        elif current_mode == 3:
            processed_frame = apply_histogram_equalization(frame)
            mode_name = "Histogram Equalization"
        elif current_mode == 4:
            processed_frame = apply_soft_blur(frame)
            mode_name = "Soft Blur"
        elif current_mode == 5:
            processed_frame = apply_cartoon_filter(frame)
            mode_name = "Cartoon Filter"
        else:
            processed_frame = frame  # No filter
            mode_name = "Original"
        
        # Display mode name on the frame
        cv2.putText(processed_frame, f"Mode: {mode_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_frame, "Press 0-5 to change mode, 'q' to quit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display the frame
        cv2.imshow('Video Filters', processed_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Process key press
        if key == ord('q'):
            break
        elif key >= ord('0') and key <= ord('5'):
            current_mode = key - ord('0')
            print(f"Switched to mode: {current_mode}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()