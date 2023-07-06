import pygame
from PIL import Image
import sys
import joblib
import pygame.camera as camera
from io import BytesIO

BASE_DIR = "./"
sys.path.insert(0, (BASE_DIR + 'Hand_Tracker'))

from HandTrackingModule import HandDetector
hand_detector = HandDetector()
classifier = joblib.load((BASE_DIR + "Hand_Tracker/Model/Predictor.pkl"))


class VideoCamera(object):
    def __init__(self):
        pygame.init()
        camera.init()
        self.camlist = camera.list_cameras()
        if not self.camlist:
            raise ValueError("No cameras found.")
        self.camera = camera.Camera(self.camlist[0], (640, 480))
        self.camera.start()

    def __del__(self):
        self.camera.stop()

    def get_frame(self):
        frame = self.camera.get_image()
        
        # Convert the captured frame to a PIL Image
        pil_image = Image.frombytes('RGB', frame.get_size(), pygame.image.tostring(frame, 'RGB'))

        # Convert the PIL Image to JPEG bytes
        with BytesIO() as output:
            pil_image.save(output, format='JPEG')
            jpeg_bytes = output.getvalue()

        return jpeg_bytes

    def get_result(self):
        frame = self.camera.get_image()
        
        # Convert the captured frame to a PIL Image
        pil_image = Image.frombytes('RGB', frame.get_size(), pygame.image.tostring(frame, 'RGB'))

        # Perform hand detection and classification
        predicted_data = hand_detector.get_result_as_dict(pil_image, classifier)

        return predicted_data

vidcam = VideoCamera()


# import sys
# import cv2
# import joblib
# import numpy as np

# from HandTrackingModule import HandDetector

# BASE_DIR = "./"
# sys.path.insert(0, (BASE_DIR + 'Hand_Tracker'))

# hand_detector = HandDetector()
# cap = cv2.VideoCapture(0)
# predictor = joblib.load((BASE_DIR + "Hand_Tracker/Model/Predictor.pkl"))

# MIN_TRACK_CON = 0.9

# while True:
#     success, img = cap.read()
#     to_print = hand_detector.get_result_as_dict(img, predictor)
#     cv2.putText(
#         img,
#         f'{to_print}',
#         (20, 70),
#         cv2.FONT_HERSHEY_PLAIN,
#         1,
#         (0, 0, 0),
#         3
#     )

#     cv2.imshow("Analyze", img)
#     cv2.waitKey(1)

# cv2.destroyAllWindows()
