import cv2
import os

# Load the eye detector
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Directory to store cropped eye images
output_dir = "detected_eyes"
os.makedirs(output_dir, exist_ok=True)

# Start video capture
cap = cv2.VideoCapture(0)

# Flags to track detected eyes
person_counter = 1  # Counter to store images for different people

# Ensure OpenCV GUI functions correctly
cv2.startWindowThread()

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    for (ex, ey, ew, eh) in eyes:
        eye_center_x = ex + ew // 2
        frame_center_x = img.shape[1] // 2

        # Determine if it's the left or right eye based on position
        if eye_center_x < frame_center_x:
            eye_label = "Left Eye"
            color = (0, 255, 0)  # Green for left eye
        else:
            eye_label = "Right Eye"
            color = (255, 0, 0)  # Blue for right eye

        # Draw a rectangle around the detected eye
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), color, 2)
        cv2.putText(img, eye_label, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with detected eyes
    cv2.imshow('Eye Detection', img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Save the eyes when 's' is pressed
        for (ex, ey, ew, eh) in eyes:
            eye_center_x = ex + ew // 2
            frame_center_x = img.shape[1] // 2

            if eye_center_x < frame_center_x:
                eye_label = "Left Eye"
                eye_region = img[ey:ey + eh, ex:ex + ew]
                cv2.imwrite(os.path.join(output_dir, f"person_{person_counter}_left_eye.jpg"), eye_region)
            else:
                eye_label = "Right Eye"
                eye_region = img[ey:ey + eh, ex:ex + ew]
                cv2.imwrite(os.path.join(output_dir, f"person_{person_counter}_right_eye.jpg"), eye_region)

        print(f"Eyes detected and saved for person {person_counter}.")
        person_counter += 1

    elif key == ord('q'):  # Quit when 'q' is pressed
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
