from PIL import Image
from tkinter.messagebox import askyesno

from preprocessing import *
from models.larger_cnn import *

def main(showPreprocessing=False):
    print("Loading model...")
    model = torch.load("models/trained_models/larger-cnn.pt", map_location=torch.device('cpu'))
    model.eval()
    print("Model loaded...")

    # Open camera (this takes a couple seconds)
    print("Turning on camera...")
    cap = cv2.VideoCapture(0)
    print("Camera is on...")

    # Ensure camera is open (this takes a couple seconds)
    print("Checking if camera is open...")
    if not cap.open(0):
        print("Error: Couldn't open camera.")
        exit(1)

    print("Starting face detection...")
    while True:

        # Read frame from camera (this takes a couple seconds on the first frame)
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

            # Convert cv2 image to PIL image
            PIL_img: Image = Image.fromarray(img)

            # Get prediction from model
            transform = getTransform(classification=True)  # Get training transform for model
            PIL_img = transform(PIL_img)  # Apply transform to PIL image before feeding to model
            output = model(PIL_img.unsqueeze(0))
            _, predicted = torch.max(output, 1)  # Get most confident prediction from model
            prediction: str = classes[predicted.item()]  # Get the label from the prediction

            # Display processed image in frame
            if showPreprocessing:
                transform = getTransform(display=True)  # Get transform for displaying image
                PIL_img = transform(img).convert("RGB")
                # Convert tensor to cv2 image
                img = np.array(PIL_img)[..., ::-1].copy()
                img = resize(img, w)
                frame[y:y + h, x:x + w, :] = img

            # Draw rectangle around face and label with emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {prediction}", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Display window
        cv2.imshow("Face Detection", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nq Key Pressed: Exiting...")
            break

if __name__ == "__main__":
    msg: str = ("Would you like the preprocessing to be overlaid on the camera feed?\n"
                "(This could be used to tune the preprocessing functions.)")
    preprocessing: bool = askyesno("Preprocessing", msg)
    main(showPreprocessing=preprocessing)
