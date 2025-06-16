import cv2
import threading
from LiveObjectDetector import LostMemeberDetector
from FaceReader import FaceDetector
# object can be any type of video
class VideoCamera(object):
    # Initialise the Video Camera for the Human and Dog detection
    def __init__(self, human=False, face=False):
        self.human = human
        self.face = face
        self.video = None
        # Threading lock helps to stabilise the program and stop video streams from overlaping each other.
        self.lock = threading.Lock()
        self.detector = LostMemeberDetector(human=self.human, camera=None)
        self.face_reader = FaceDetector(camera=None)

    def __del__(self):
        if self.video and self.video.isOpened():
            self.video.release()

    def set_mode(self, human, face):
        with self.lock:
            print(f"Changing mode to human={human}, face={face}")
            self.human = human
            self.face = face
            self.detector.human = human

    def get_frame(self):
        processed_frame = None
        # This function gets the frame fed to it and processes it based on if human/face/dog detection is selected
        # The locking of the stream ensure the image reading and processing is kept stable.
        with self.lock:
            if self.video is None or not self.video.isOpened():
                print(f"Attempting to open camera at index 1")
                # CAP_DSHOW for better compatibility with virtual cameras
                self.video = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                if not self.video.isOpened():
                    print(f"Error: Could not open camera at index 1.")
                    # Return an empty byte string if connection is not established
                    return b''
            # Reads frame by frame for object detection
            success, frame = self.video.read()
            if not success:
                return b''
            # Chooses which detector based on the user's selection for face or body/dog detection
            if self.face:
                processed_frame = self.face_reader.process_face(frame)
            else:
                processed_frame = self.detector.process_frame(frame)

            if processed_frame is None:
                return b''

            ret, jpeg = cv2.imencode('.jpg', processed_frame)
            if not ret:
                return b''
            # Camera is opened correctly and we can read from it
            success, frame = self.video.read()

            if not success:
                # If reading fails, return empty bytes
                return b''

            # Get the processed frame from the appropriate detector
            if self.face:
                processed_frame = self.face_reader.return_face(frame)
            else:
                processed_frame = self.detector.return_frame(frame)

        # Final safety check before encoding
        if processed_frame is None:
            return b''

        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        if not ret:
            return b''

        return jpeg.tobytes()

