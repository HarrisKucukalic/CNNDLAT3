import cv2
from DogDetectorLive import LostMemeberDetector
# object can be any type of video
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # ret, frame = self.video.read()
        # # convert to jpeg, ret is a flag to check if anything is wrong
        # ret, jpeg = cv2.imencode('.jpg', frame)
        # # send bytes to browser
        # return jpeg.tobytes()
        detector = LostMemeberDetector(human=True, camera=cv2.VideoCapture(0))
        frame = detector.return_frame()
        ret, jpeg = cv2.imencode('.jpg', frame)
        # send bytes to browser
        return jpeg.tobytes()
