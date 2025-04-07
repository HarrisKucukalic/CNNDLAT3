import cv2
from LiveObjectDetector import LostMemeberDetector
from FaceReader import FaceDetector
# object can be any type of video

class VideoCamera(object):
    def __init__(self, human=False):
        self.human = human
        self.video = cv2.VideoCapture(0)
        self.detector = LostMemeberDetector(human=self.human, camera=self.video)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        frame = self.detector.return_frame()
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
