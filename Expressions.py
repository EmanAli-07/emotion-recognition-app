import cv2
from fer import FER
import streamlit as st
import numpy as np

# Initialize the FER detector
detector = FER()

# Function to process webcam input and detect emotions
def detect_emotions(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    result = detector.detect_emotions(rgb_frame)
    
    if result:
        for face in result:
            (x, y, w, h) = face['box']
            emotions = face['emotions']
            emotion = max(emotions, key=emotions.get)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Streamlit interface
st.title("Real-Time Emotion Recognition")

# Ask user if they want to start emotion detection
start_detection = st.button("Start Emotion Detection")

if start_detection:
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Check if webcam is accessible
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        # Start detection
        st.write("Press 'q' to stop the webcam.")
        
        # Stream video to Streamlit
        while True:
            ret, frame = cap.read()
            
            if not ret:
                st.error("Error: Could not read frame.")
                break
            
            # Detect emotions in the frame
            frame = detect_emotions(frame)

            # Convert the frame to a format Streamlit can display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, channels="RGB")

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam
        cap.release()
        cv2.destroyAllWindows()
