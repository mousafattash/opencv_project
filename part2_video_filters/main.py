import cv2

def apply_edge_detection(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    
    # Convert back to BGR for display
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return edges_colored

def apply_cartoon_filter(frame):
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

def apply_grayscale_quantization(frame, levels=8):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply quantization
    # Formula: quantized = (gray // (256 // levels)) * (256 // levels)
    quantization_factor = 256 // levels
    quantized = (gray // quantization_factor) * quantization_factor
    
    # Convert back to BGR for display
    return cv2.cvtColor(quantized, cv2.COLOR_GRAY2BGR)

def apply_histogram_equalization(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    # Convert back to BGR for display
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def apply_soft_blur(frame, kernel_size=15):
    # Apply Gaussian blur for a soft appearance
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    # For a more aesthetic soft glow effect, we can blend the original with the blurred
    soft_glow = cv2.addWeighted(frame, 0.3, blurred, 0.7, 0)
    
    return soft_glow

def apply_clahe(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE
    equalized = clahe.apply(gray)
    
    # Convert back to BGR for display
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def display_menu():
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