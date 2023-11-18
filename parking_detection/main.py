import cv2
import pickle
import cvzone
import numpy as np

# Video feed from cameras
cap = cv2.VideoCapture('./footage/carPark.mp4')

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

free_space_color = (0, 255, 0) 
occupied_space_color = (0, 0, 255) 

def checkParkingSpace(imgPro):
    spaceCounter = 0
    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]

        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = free_space_color
            thickness = 5
            spaceCounter += 1
        else:
            color = occupied_space_color
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)

        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1, thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=free_space_color)

legend_img = np.ones((150, 300, 3), dtype=np.uint8) * 255  # White background
cv2.putText(legend_img, 'Color Legend', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.rectangle(legend_img, (10, 60), (40, 90), free_space_color, -1)
cv2.putText(legend_img, 'Free Space', (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.rectangle(legend_img, (10, 100), (40, 130), occupied_space_color, -1)
cv2.putText(legend_img, 'Occupied Space', (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.imshow("Color Legend", legend_img)

while True:
    # Check if the video has reached its end, reset if true
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read a frame from the video feed
    success, img = cap.read()

    # Convert the frame to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)

    # Apply adaptive thresholding to create a binary image
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

    imgMedian = cv2.medianBlur(imgThreshold, 5)

    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    checkParkingSpace(imgDilate)

    cv2.imshow("Image", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow("Color Legend")
cv2.destroyAllWindows()