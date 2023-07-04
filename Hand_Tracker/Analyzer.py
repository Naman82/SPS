import cv2
import xgboost
import sklearn
import joblib
import numpy as np

from HandTrackingModule import HandDetector

hand_detector = HandDetector()
cap = cv2.VideoCapture(0)
predictor = joblib.load("E:\Coding\Placement Projects\SPS\Hand_Tracker\Model\Predictor (1).pkl")
MIN_TRACK_CON = 0.9

while True:
    success, img = cap.read()
    to_print = hand_detector.get_result_as_dict(img, predictor)
    cv2.putText(
        img,
        f'{to_print}',
        (20, 70),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 0, 0),
        3
    )

    cv2.imshow("Analyze", img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
