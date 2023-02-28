from cvzone.ClassificationModule import Classifier
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)
address = 'http://192.168.1.4:8080/video'
cap.open(address)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,  # hands object declaration
    min_tracking_confidence=0.5)

labels = ["A", "B", "C"]

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=True)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("Cropped hands", imgCrop)
        cv2.imshow("WHITE", imgWhite)

    cv2.imshow("Data Collection Panel", img)
    cv2.waitKey(1)
