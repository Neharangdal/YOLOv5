import cv2
import os
import numpy as np

# Directories for input and output
input_dir = r"C:\Users\nehad\OneDrive\Desktop\yolov5\detected_eyes"
output_dir = r"C:\Users\nehad\OneDrive\Desktop\yolov5\enhanced_eyes"
os.makedirs(output_dir, exist_ok=True)

def remove_specular_reflection(image):
    """
    Removes specular reflections from the eye image using inpainting.
    """
    _, mask = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted_image

def enhance_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error reading {image_path}")
        return

    # Step 1: Remove Specular Reflections
    no_reflection_image = remove_specular_reflection(image)

    # Step 2: Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(no_reflection_image)

    # Step 3: Bilateral Filtering
    bilateral_filtered_image = cv2.bilateralFilter(clahe_image, d=15, sigmaColor=100, sigmaSpace=100)

    # Step 4: Sharpening
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(bilateral_filtered_image, -1, sharpening_kernel)

    cv2.imwrite(output_path, sharpened_image)

# Process images in detected_eyes folder
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.png')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        enhance_image(input_path, output_path)
        print(f"Enhanced image saved to {output_path}")

print("✅ All images enhanced and saved in the 'enhanced_eyes' directory.")
