import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model


# Load models.
emotion_model = load_model("models/XCEPTION.72.model")
face_detection_model = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    # Create a small pause between samples.
    time.sleep(0.5)

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get the coordinates of the faces.
    faces = face_detection_model.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) != 0:
        # Get the coordinates of the first face detected.
        face_coords = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face_coords

        # Crop the image to the coordinates of the face.
        cropped_face = frame[fY:fY + fH, fX:fX + fW]

        # Resize, black-and-white, reshape, and normalize  face.
        cropped_face = cv2.resize(cropped_face, (48, 48))
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        cropped_face = cropped_face.reshape(48, 48, 1)
        cropped_face = cropped_face / 255

        # Get the predictions (a vector of length 7) and map it to the appropriate emotion.
        preds = emotion_model.predict(np.array([cropped_face,cropped_face]))[0]
        emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
        max_emotion = np.argmax(preds)
        prediction = emotions[max_emotion]

        # Draw a boundingbox around the face with the prediction.
        cv2.putText(img=frame,
                    text=prediction, 
                    org=(fX, fY - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2)
        cv2.rectangle(img=frame,
                      pt1=(fX, fY),
                      pt2=(fX + fW, fY + fH),
                      color=(255, 255, 255),
                      thickness=2)

    # Convert the webcam's frame to black-and white.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame.
    cv2.imshow("frame", gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture.
cap.release()
cv2.destroyAllWindows()
