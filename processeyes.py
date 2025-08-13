import torch
import cv2
import os

# Load your trained YOLOv5 model
model_path = "C:\\Users\\nehad\\OneDrive\\Desktop\\yolov5\\runs\\train\\exp\\weights\\best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Paths for enhanced left and right eye images
enhanced_images_dir = "C:\\Users\\nehad\\OneDrive\\Desktop\\yolov5\\enhanced_eyes"
output_dir = "C:\\Users\\nehad\\OneDrive\\Desktop\\yolov5\\processed_eyes"
os.makedirs(output_dir, exist_ok=True)

# Class indices for iris and pupil (update according to your dataset)
IRIS_CLASS = 0  # Example: Class ID for iris
PUPIL_CLASS = 1  # Example: Class ID for pupil

# Function to process an image and calculate the ratio
def process_eye(image_path, eye_label):
    try:
        results = model(image_path)  # Updated syntax for YOLOv5 inference
        img = cv2.imread(image_path)

        iris_width = None
        pupil_width = None

        # Process detection results
        for result in results.xyxy[0]:  # Access bounding boxes directly
            x_min, y_min, x_max, y_max, conf, cls = map(int, result[:6])
            width = x_max - x_min

            # Check class and assign width
            if cls == IRIS_CLASS:
                iris_width = width
                color = (255, 0, 0)  # Blue for iris
            elif cls == PUPIL_CLASS:
                pupil_width = width
                color = (0, 255, 0)  # Green for pupil

            # Draw bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # Compute and display ratio in console
        ratio_text = f"{eye_label} Ratio: Not Detected"
        if iris_width and pupil_width:
            ratio = iris_width / pupil_width
            ratio_text = f"{eye_label} IP Ratio: {ratio:.2f}"
            print(f"{os.path.basename(image_path)}: {ratio_text}")
        else:
            print(f"{os.path.basename(image_path)}: {ratio_text}")

        # Save the processed image with bounding boxes
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, img)
        print(f"Processed image saved to: {output_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Main code block for VS Code compatibility
if __name__ == "__main__":
    # Process all enhanced images for left and right eyes
    for filename in os.listdir(enhanced_images_dir):
        filepath = os.path.join(enhanced_images_dir, filename)
        if "left_eye" in filename.lower():
            process_eye(filepath, "Left Eye")
        elif "right_eye" in filename.lower():
            process_eye(filepath, "Right Eye")
