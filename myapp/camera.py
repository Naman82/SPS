# import cv2
# import sys
# import joblib

# BASE_DIR = "./"
# sys.path.insert(0, (BASE_DIR + 'Hand_Tracker'))

# from Hand_Tracker.HandTrackingModule import HandDetector
# hand_detector = HandDetector()
# classifier = joblib.load((BASE_DIR + "Hand_Tracker/Model/Predictor.pkl"))


# class VideoCamera(object):
#     def __init__(self):
#         self.video=cv2.VideoCapture(0)

#     def __del__(self):
#         self.video.release()

#     def get_frame(self):
#         success,image=self.video.read()
#         ret,jpeg=cv2.imencode('.jpg',image)
#         return jpeg.tobytes()


#     def get_result(self):
#         success, image = self.video.read()
#         predicted_data = hand_detector.get_result_as_dict(image, classifier)
        
#         return predicted_data

import pygame
from PIL import Image
import sys
import joblib
import pygame.camera as camera
from io import BytesIO

BASE_DIR = "./"
sys.path.insert(0, (BASE_DIR + 'Hand_Tracker'))

from Hand_Tracker.HandTrackingModule import HandDetector
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
