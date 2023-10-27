from face_detection import FaceDetector

if __name__ == "__main__":
    face_cascade_path = "./haarcascade/haarcascade_frontalface_default.xml"

    detector = FaceDetector(face_cascade_path)
    detector.process_image("./Images/catchmeifyoucan.png", "age-face_detected.png")