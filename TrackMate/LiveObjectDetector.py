import cv2
import random
from ultralytics import YOLO
import torch
import torchvision

class LostMemeberDetector:
    # Assigns the models to be used for each type of detection.
    def __init__(self, human=True, yolo_dog_model='best.pt', yolo_human_model='yolo12n.pt', camera=None):
        self.cap = camera
        self.yolo_dog = YOLO(yolo_dog_model)
        self.yolo_human = YOLO(yolo_human_model)
        self.detected_classes = set()
        self.class_colours = {}
        self.human = human

    def generate_rand_col(self, class_name):
        # Generate a unique color for each class if not already assigned.
        if class_name not in self.class_colours:
            self.class_colours[class_name] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
        return self.class_colours[class_name]

    def process_frame(self, img):
        if img is None:
            return None

        detections = []
        if self.human:
            # Process frame with TrackMateTests model (only keep 'person' class)
            human_results = self.yolo_human(img, device=0)
            for result in human_results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = result.names[class_id]
                    confidence = box.conf[0].item()  # Confidence score

                    if class_name.lower() == 'person':  # Only draw 'person'
                        colour = self.generate_rand_col(class_name)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append((class_name, confidence, x1, y1, x2, y2, colour))

        # if it's not a person that is being looked for, we default to our custom dog breed detector.
        else:
            highest_conf_breed = None
            highest_conf = 0

            dog_results = self.yolo_dog(img, device=0)
            for result in dog_results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = result.names[class_id]
                    confidence = box.conf[0].item()

                    if confidence > highest_conf:
                        highest_conf = confidence
                        highest_conf_breed = (class_name, confidence, box.xyxy[0])

            # Add the highest-confidence TrackMate breed to detections
            if highest_conf_breed:
                class_name, confidence, box_coords = highest_conf_breed
                colour = self.generate_rand_col(class_name)
                x1, y1, x2, y2 = map(int, box_coords)
                detections.append((class_name, confidence, x1, y1, x2, y2, colour))

        # Draw detections where the confidence of the dog breed is 0.6. This reduces the probability of incorrect predictions and multi dog breed detection
        for class_name, confidence, x1, y1, x2, y2, colour in detections:
            if confidence > 0.6:
                cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        return img

    # As in the face reader, these components below are for the front end static display of classified images.
    def get_image_prediction(self, img):
        detections = []
        if self.human:
            results = self.yolo_human(img, device=0)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = result.names[class_id]
                    confidence = box.conf[0].item()
                    if class_name.lower() == 'person' and confidence > 0.6:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        colour = self.generate_rand_col(class_name)
                        detections.append((class_name, confidence, x1, y1, x2, y2, colour))
        else:
            results = self.yolo_dog(img, device=0)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = result.names[class_id]
                    confidence = box.conf[0].item()
                    if confidence > 0.6:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        colour = (255, 0, 0)
                        detections.append((class_name, confidence, x1, y1, x2, y2, colour))

        # Draw detections
        for class_name, confidence, x1, y1, x2, y2, colour in detections:
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        return img

    def return_frame(self, frame):
        return self.process_frame(frame)