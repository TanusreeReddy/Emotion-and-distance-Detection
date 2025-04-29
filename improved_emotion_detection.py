import cv2
from deepface import DeepFace
import numpy as np
from datetime import datetime

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Check if the cascade classifier loaded correctly
if face_cascade.empty():
    print("Error: Couldn't load face cascade classifier.")
    exit()

# Start video capture (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Couldn't open webcam.")
    exit()

# Get frame dimensions and FPS for video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20  # Standard FPS for video output

# Initialize video writer
output_filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Check if video writer initialized correctly
if not out.isOpened():
    print("Error: Couldn't initialize video writer.")
    cap.release()
    exit()

# Constants for distance calculation
KNOWN_FACE_WIDTH = 14.0  # Average face width in cm
FOCAL_LENGTH = 714.29    # Replace with your calculated focal length

# Store emotion history for each face (keyed by approximate face position)
face_emotions = {}
HISTORY_SIZE = 3  # Number of frames to average emotions over

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with parameters tuned for multiple faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=10,
        minSize=(60, 60)
    )

    # Update face emotions dictionary
    current_faces = []
    for (x, y, w, h) in faces:
        # Use face position as a simple key (center of the face)
        face_key = (x + w // 2, y + h // 2)

        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Calculate distance
        face_width_pixels = w
        if face_width_pixels > 0:
            distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / face_width_pixels
            distance_text = f"{distance:.1f} cm"
        else:
            distance_text = "Unknown"

        # Preprocess the face: resize and convert to RGB
        try:
            face_resized = cv2.resize(face, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Detect emotion
            result = DeepFace.analyze(
                face_rgb,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )
            emotion = result[0]["dominant_emotion"]

            # Initialize or update emotion history for this face
            if face_key not in face_emotions:
                face_emotions[face_key] = []
            face_emotions[face_key].append(emotion)
            if len(face_emotions[face_key]) > HISTORY_SIZE:
                face_emotions[face_key].pop(0)

            # Get the most common emotion for this face
            most_common_emotion = max(set(face_emotions[face_key]), key=face_emotions[face_key].count)

        except Exception:
            most_common_emotion = "Unknown"

        # Draw bounding box and labels (emotion and distance)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display emotion (Black text, white background)
        cv2.putText(
            frame,
            most_common_emotion,
            (x, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),  # Black text
            2,
            cv2.LINE_AA
        )
        text_size_emotion, _ = cv2.getTextSize(most_common_emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x, y - 55), (x + text_size_emotion[0], y - 25), (255, 255, 255), -1)
        cv2.putText(
            frame,
            most_common_emotion,
            (x, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),  # Black text
            2,
            cv2.LINE_AA
        )

        # Display distance (Black text, white background)
        text_size_distance, _ = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x, y - 25), (x + text_size_distance[0], y + 5), (255, 255, 255), -1)
        cv2.putText(
            frame,
            distance_text,
            (x, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),  # Black text
            2,
            cv2.LINE_AA
        )

        # Track current faces
        current_faces.append(face_key)

    # Clean up old faces no longer detected
    face_emotions = {k: v for k, v in face_emotions.items() if k in current_faces}

    # Write frame to video
    out.write(frame)

    # Show frame
    cv2.imshow("Emotion and Distance Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
out.release()
cap.release()
cv2.destroyAllWindows()
