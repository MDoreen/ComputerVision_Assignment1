import cv2
import numpy as np

class FaceDetector:
    def __init__(self, face_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def detect_faces(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def estimate_age(self, face):
        # estimate age based on face size
        face_height, face_width = face.shape[:2]
        estimated_age = int((face_height + face_width) / 4)
        return estimated_age

    def process_image(self, image_path, output_path):
        faces = self.detect_faces(image_path)
        image = cv2.imread(image_path)  
        for (x, y, w, h) in faces:
            face = cv2.resize(image[y:y+h, x:x+w], (227, 227))
            age = self.estimate_age(face)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"Age: {age} years", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Faces Detected", image)
        cv2.imwrite(output_path, image)  # Save the image with bounding boxes and age predictions
        cv2.waitKey(0)
        cv2.destroyAllWindows()