import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Load in color first
image = cv2.imread('part1_image_processing/input/mara3i.jpg')
if image is None:
    raise Exception("Image not found!")

# Generate random watermark position (within the image bounds)
x = random.randint(0, image.shape[1] - 200)  # Keep watermark within image width
y = random.randint(0, image.shape[0] - 50)   # Keep watermark within image height

# Add watermark text to the image (black color for a bright image)
watermark_text = "Ahmad Khalil - 12027692"
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, watermark_text, (x, y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

# Save the image with watermark 
cv2.imwrite("watermarked_image.jpg", image)

# Convert image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("grayscale_image.jpg", image_gray)

# Calculate mean, min, max of grayscale image
mean_val = np.mean(image_gray)
min_val = np.min(image_gray)
max_val = np.max(image_gray)

# Show image shape 
print("Grey Image height, width:", image_gray.shape)  # (height, width)

# Grayscale image has 1 channel (since it's grayscale)
if len(image_gray.shape) == 2:
    print("Grayscale Color Channels: 1")
else:
    print(f"Grayscale Color Channels: {image_gray.shape[2]}")

print("Mean Value:", mean_val, "Min Value:", min_val, "Max Value:", max_val)

# Apply random brightness modification
c = round(random.uniform(0.4, 2.0), 2)
print("Random Brightness Coefficient (c):", c)

# Apply brightness modification
image_bright = image_gray.astype(np.float32) * c

# Clip to valid pixel range before saving and plotting
image_bright = np.clip(image_bright, 0, 255).astype(np.uint8)
# Save the result 
cv2.imwrite("brightness_modified.jpg", image_bright)

# Display comparison between 3 images: original watermarked, grayscale, and brightness modified
# plt.figure(figsize=(15, 5))

# Show original watermarked image
# plt.subplot(1, 3, 1)
# plt.title("Watermarked Image")
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# Show grayscale image
# plt.subplot(1, 3, 2)
# plt.title("Grayscale Image")
# plt.imshow(image_gray, cmap='gray')
# plt.axis('off')

# Show brightness-modified image
# plt.subplot(1, 3, 3)
# plt.title(f"Brightness Modified (c={c})")
# plt.imshow(image_bright, cmap='gray')
# plt.axis('off')
#plt.show()

plt.hist(image_bright.flatten(), bins=256, range=(0, 255))  
plt.title("Histogram of the Modified-brightness Image")
# plt.show()

# Normalize to [0, 255] range using OpenCV normalize function
image_normalized = cv2.normalize(image_bright, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("normalized_image.jpg", image_normalized)



image_equalized = cv2.equalizeHist(image_bright)
cv2.imwrite("equalized_image.jpg", image_equalized)

mean_intensity = np.mean(image_equalized)
gamma = np.log10(0.5) / np.log10(mean_intensity / 255)
image_gamma = np.power(image_bright / 255.0, gamma) * 255
image_gamma = np.clip(image_gamma, 0, 255).astype(np.uint8)
cv2.imwrite("gamma_corrected.jpg", image_gamma)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(image_gray.flatten(), bins=256, range=(0, 255))
plt.title("Histogram of Grey Image")



#Create two subplots side by side
plt.subplot(1, 2, 2)
plt.hist(image_equalized.flatten(), bins=256, range=(0, 255))
plt.title("Histogram of the corrected Image")



plt.tight_layout()  # Adjust the spacing between subplots
plt.show()