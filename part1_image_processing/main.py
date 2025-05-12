import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Load in color first
image = cv2.imread('part1_image_processing/input.jpg')
if image is None:
    raise Exception("Image not found!")

# Add watermark text to the image (black color for a bright image)
watermark_text = "Ahmad Khalil - 12027692"
font = cv2.FONT_HERSHEY_SIMPLEX
(text_width, text_height), _ = cv2.getTextSize(watermark_text, font, 1, 2)

# Compute max x and y where the text can be placed without going out of bounds
max_x = image.shape[1] - text_width
max_y = image.shape[0] - text_height

# Generate random position
x = random.randint(0, max_x)
y = random.randint(text_height, image.shape[0])  # ensure y is below top and above bottom

# Put the text at the random position
cv2.putText(image, watermark_text, (x, y), font, 1, (0, 0, 0), 2)

cv2.imwrite("watermarked_image.jpg", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("grayscale_image.jpg", image_gray)

# Calculate mean, min, max of grayscale image
mean_val = np.mean(image_gray)
min_val = np.min(image_gray)
max_val = np.max(image_gray)

print("Grey Image height, width:", image_gray.shape)  # (height, width)

if len(image_gray.shape) == 2:
    print("Grayscale Color Channels: 1")
else:
    print(f"Grayscale Color Channels: {image_gray.shape[2]}")

print("Mean Value:", mean_val, "Min Value:", min_val, "Max Value:", max_val)


c = 1.63
print("Random Brightness Coefficient (c):", c)
# Apply brightness modification
image_bright = image_gray.astype(np.float32) * c
image_bright = np.clip(image_bright, 0, 255).astype(np.uint8)
cv2.imwrite("brightness_modified.jpg", image_bright)
plt.hist(image_bright.flatten(), bins=256, range=(0, 255))  
plt.title("Histogram of the Modified-brightness Image")

# Apply linear contrast stretching
min_val_bright = np.min(image_bright)
max_val_bright = np.max(image_bright)
if max_val_bright > min_val_bright:
    image_stretched = (image_bright - min_val_bright) * (255.0 / (max_val_bright - min_val_bright))
else:
    image_stretched = image_bright.copy()
image_stretched = np.clip(image_stretched, 0, 255).astype(np.uint8)

mean_pixel_value = np.mean(image_gray)
dark_threshold = 120
if mean_pixel_value > dark_threshold:
    gamma = 1.5
    image_gamma_dark = np.uint8(np.clip(c * np.power(image_gray / 255.0, gamma) * 255.0, 0, 255))
    corrected_image = image_gamma_dark
else:
    corrected_image = image_stretched
cv2.imwrite("corrected_image.jpg", corrected_image)

# Display comparison between 3 images: original watermarked, grayscale, and brightness modified
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Grayscale Image")
plt.imshow(cv2.cvtColor(image_gray, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("brightness-modified Image")
plt.imshow(image_bright, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f"corrected_image (c={c})")
plt.imshow(corrected_image, cmap='gray')
plt.axis('off')
plt.show()
#---------------------------------------------------------------------------------------
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(image_gray.flatten(), bins=256, range=(0, 255))
plt.title("Histogram of Grey Image")

#Create two subplots side by side
plt.subplot(1, 2, 2)
plt.hist(corrected_image.flatten(), bins=256, range=(0, 255))
plt.title("Histogram of the corrected Image")

plt.tight_layout()  # Adjust the spacing between subplots
plt.show()

# Amount of noise (e.g., 2% noise)
amount = 0.02
# Create a copy of the image
noisy = image_stretched.copy()
# Add salt (white pixels)
salt = np.random.random(image_stretched.shape) < amount / 2
noisy[salt] = 255
# Add pepper (black pixels)
pepper = np.random.random(image_stretched.shape) < amount / 2
noisy[pepper] = 0
# Save the noisy image
cv2.imwrite("noisy_image.jpg", noisy)

mean_filtered = cv2.blur(noisy, (3, 3))
cv2.imwrite("noisy_mean_image.jpg", mean_filtered)

bilateral_filtered = cv2.bilateralFilter(mean_filtered, d=9, sigmaColor=75, sigmaSpace=75)
cv2.imwrite("bilateral_filtered.jpg", bilateral_filtered)

median_filtered = cv2.medianBlur(noisy, 3)
cv2.imwrite("noisy_median_image.jpg", median_filtered)