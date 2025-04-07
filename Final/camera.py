import cv2
from LiveObjectDetector import LostMemeberDetector
from FaceReader import FaceDetector
# object can be any type of video

class VideoCamera(object):
    def __init__(self, human=False, face=False):
        self.human = human
        self.face = face
        self.video = cv2.VideoCapture(0)
        self.detector = LostMemeberDetector(human=self.human, camera=self.video)
        if face and human:
            self.face_reader = FaceDetector(camera=self.video)

    def __del__(self):
        self.video.release()


    def get_frame(self):
        if self.face:
            frame = self.face_reader.return_face()
        elif not self.face:
            frame = self.detector.return_frame()
        else:
            ret, frame = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Return the JPEG-encoded image as bytes
        return jpeg.tobytes()