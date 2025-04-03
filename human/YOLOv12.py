import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO
from colorama import Fore, Style

class LostMemeberDetector:
    def __init__(self, yolo_model='yolo12n.pt', max_faces=10, classes_to_label={'person'}):
        self.cap = cv2.VideoCapture(0)
        self.detector = FaceMeshDetector(maxFaces=max_faces)
        self.yolo = YOLO(yolo_model)
        self.detected_classes = set()
        self.face_marks_list = []
        self.classes_to_label = classes_to_label

    def process_frame(self):
        success, img = self.cap.read()
        if not success:
            return None

        # YOLO Object Detection
        results = self.yolo(img)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = result.names[class_id]

                if class_name.lower() in self.classes_to_label:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    self.detected_classes.add(class_name)

        # Face Mesh Detection
        if 'person' in self.classes_to_label:
            img, faces = self.detector.findFaceMesh(img)
            if faces:
                for face in faces:
                    self.face_marks_list.append(face)
                    pointLeft = face[145]
                    pointRight = face[374]
                    w, _ = self.detector.findDistance(pointLeft, pointRight)
                    W = 6.3
                    # manually calculated focal length of camera, has to be re-calibrated.
                    f = 800
                    d = (W * f) / w
                    scale = max(1.5, min(2, d / 100))
                    cvzone.putTextRect(img, f'Distance: {int(d)}cm', (face[10][0] - 100, face[10][1] - 50), scale=scale)

        return img

    def run(self):
        while True:
            img = self.process_frame()
            if img is None:
                break

            cv2.imshow('YOLO + Face Mesh', img)

            if cv2.waitKey(1) & (cv2.getWindowProperty('YOLO + Face Mesh', cv2.WND_PROP_VISIBLE) < 1):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Example usage when running this file directly
if __name__ == "__main__":
    detector = LostMemeberDetector(classes_to_label={'person'})
    detector.run()
