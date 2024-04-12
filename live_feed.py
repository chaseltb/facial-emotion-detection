from PIL import Image

from preprocessing import *
from models.larger_cnn import *

def main():
    print("Loading model...")
    model = torch.load("models/trained_models/larger-cnn.pt", map_location=torch.device('cpu'))
    model.eval()
    print("Model loaded...")

    print("Turning on camera...")
    cap = cv2.VideoCapture(0)
    print("Camera is on...")

    # if not cap.open(0):
    #     print("Error: Couldn't open camera.")
    #     exit(1)

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
            img = pipline(frame[y:y+h, x:x+w])

            PIL_img: Image = Image.fromarray(img)
            transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float),
                                            transforms.Grayscale(), transforms.Normalize(0.5, 0.5)])
            PIL_img = transform(PIL_img)
            output = model(PIL_img.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            prediction: str = classes[predicted.item()]

            # Draw face on frame
            img = resize(img, w)
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            frame[y:y + h, x:x + w, :] = img
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {prediction}", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Display frame
        cv2.imshow("Face Detection", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nQ Key Pressed: Exiting...")
            break

if __name__ == "__main__":
    main()
