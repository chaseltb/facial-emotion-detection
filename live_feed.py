import cv2

from preprocessing import *

def main():
    print("Turning on camera...")
    cap = cv2.VideoCapture(0)
    print("Camera is on...")

    if not cap.open(0):
        print("Error: Couldn't open camera.")
        exit(1)

    print("Starting face detection...")
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read frame.")
            break

        grey = to_grayscale(frame)
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        faces = face_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            img = pipline(frame[y:y+h, x:x+w, :])
            img = resize(img, w)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            frame[y:y+h, x:x+w, :] = img
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 4)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
