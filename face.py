import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN  # More accurate face detection

# Load the trained anti-spoofing model
model = load_model(r"C:\Users\21311\Major Project1\face_anti_spoofing_model.h5")

# Initialize MTCNN for face detection
detector = MTCNN()

# Function to preprocess the face image
def preprocess_face(face):
    face = cv2.resize(face, (224, 224))  # Resize to match model input size
    face = face / 255.0  # Normalize pixel values
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Open the laptop camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using MTCNN
    faces = detector.detect_faces(frame)

    # Loop through detected faces
    for result in faces:
        x, y, w, h = result['box']
        confidence = result['confidence']

        # Only process faces with high confidence
        if confidence > 0.9:
            # Extract the face region
            face = frame[y:y+h, x:x+w]

            # Preprocess the face for the anti-spoofing model
            processed_face = preprocess_face(face)

            # Predict if the face is real or spoofed
            prediction = model.predict(processed_face)
            print(f"Prediction score: {prediction[0][0]}")  # Debugging info

            # Adjust threshold if needed
            threshold = 0.5
            label = 'real' if prediction > threshold else 'spoof'
            confidence_score = float(prediction[0][0]) if label == 'real' else 1 - float(prediction[0][0])

            # Draw bounding box and label on the frame
            color = (0, 255, 0) if label == 'real' else (0, 0, 255)  # Green for real, red for spoof
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'{label} ({confidence_score:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('Face Anti-Spoofing', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()