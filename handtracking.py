import cv2
import mediapipe as mp

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize camera (0 for default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break
    
    # Flip the frame horizontally for a more intuitive mirror view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe requires RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    result = hands.process(frame_rgb)
    
    # If hands are detected, draw landmarks on them
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the frame with landmarks overlaid
    cv2.imshow('Hand Tracking', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
