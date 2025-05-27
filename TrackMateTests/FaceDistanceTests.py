import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=10)


while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img)

    if faces:
        for face in faces:
            pointLeft = face[145]
            pointRight = face[374]
            w, _ = detector.findDistance(pointLeft, pointRight)
            W = 6.3

            # Finding distance
            f = 800
            d = (W*f)/w

            scale = max(1.5, min(2, d / 100))  # Adjust scale based on distance, with limits

            cvzone.putTextRect(img, f'Distance: {int(d)}cm', (face[10][0] - 100, face[10][1] - 50), scale=scale)

    cv2.imshow("Image", img)
    # Check if 'X' button is clicked or 'q' key is pressed
    if cv2.waitKey(1) & (cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1):
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()







