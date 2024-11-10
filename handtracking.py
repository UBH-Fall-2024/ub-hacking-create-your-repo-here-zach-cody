import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set camera resolution (optional but may improve detection)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Define a broader HSV range for neon pink color
lower_pink = np.array([130, 100, 100], dtype=np.uint8)
upper_pink = np.array([170, 255, 255], dtype=np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for neon pink based on the defined HSV range
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Apply morphological operations to close gaps and remove small noise
    kernel = np.ones((3, 3), np.uint8)  # Smaller kernel size
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the pink mask
    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track the biggest contour (the stick)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Lowered threshold
            # Get the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw the bounding box around the detected stick
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add label "Stick" to the bounding box
            cv2.putText(frame, "Stick", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the result with the detected neon pink stick
    cv2.imshow('Neon Pink Stick Tracking', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
