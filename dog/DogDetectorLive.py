import cv2
import random
from ultralytics import YOLO
import torch
import torchvision

class LostMemeberDetector:
    def __init__(self, human=False, yolo_dog_model='best.pt', yolo_human_model='yolo12n.pt'):
        self.cap = cv2.VideoCapture(0)
        self.yolo_dog = YOLO(yolo_dog_model)
        self.yolo_human = YOLO(yolo_human_model)
        self.detected_classes = set()
        self.class_colours = {}
        self.human = human

    def generate_rand_col(self, class_name):
        """Generate a unique color for each class if not already assigned."""
        if class_name not in self.class_colours:
            self.class_colours[class_name] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
        return self.class_colours[class_name]

    def process_frame(self):
        success, img = self.cap.read()
        if not success:
            return None

        detections = []
        if self.human:
            # Process frame with human model (only keep 'person' class)
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

            # Add the highest-confidence dog breed to detections
            if highest_conf_breed:
                class_name, confidence, box_coords = highest_conf_breed
                colour = self.generate_rand_col(class_name)
                x1, y1, x2, y2 = map(int, box_coords)
                detections.append((class_name, confidence, x1, y1, x2, y2, colour))

        # Draw all detections
        for class_name, confidence, x1, y1, x2, y2, colour in detections:
            if confidence > 0.6:
                cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        return img

    def run(self):
        while True:
            img = self.process_frame()
            if img is None:
                break

            cv2.imshow('Detection Live', img)

            if cv2.waitKey(1) & (cv2.getWindowProperty('Detection Live', cv2.WND_PROP_VISIBLE) < 1):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Example usage when running this file directly
if __name__ == "__main__":
    print("Torch Version:", torch.__version__)
    print("TorchVision Version:", torchvision.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device:", torch.cuda.get_device_name(0))
    print("Settings")
    detector = LostMemeberDetector(human=False)
    detector.run()
