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

        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            continue

        # Identify faces in frame
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        # Preprocess and draw onto frame
        for (x, y, w, h) in faces:
            # Preprocess image
            img = pipline(frame[y:y+h, x:x+w, :])
            img = resize(img, w)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Draw face on frame
            frame[y:y+h, x:x+w, :] = img
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 4)

        # Display frame
        cv2.imshow("Face Detection", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
