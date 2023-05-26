import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Load the video file
cap = cv2.VideoCapture("0327.mp4")

# Create a pose estimator
pose = mp.solutions.pose.Pose()

# Start the video capture loop
while True:
    # Get the current frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    # Estimate the pose of the person in the current frame
    results = pose.process(frame)

    # Draw the pose of the person on the current frame
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x, y, z = landmark.x, landmark.y, landmark.z
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Display the current frame
    cv2.imshow("Frame", frame)

    # Check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Close the video capture object
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()