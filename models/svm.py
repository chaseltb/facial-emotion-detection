import pickle
import os

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA


DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "../data")
CATEGORIES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
LABELS_FILE = "..\\data\\labels.csv"
TEST_PERCENT = 0.2
MODEL_SAVE_DIRECTORY = os.path.join(os.path.dirname(__file__), "../models/trained_models/svmModel.pkl")


def create_datasets():
    # Bring in the preprocessed images
    labels = pd.read_csv(os.path.join(os.path.dirname(__file__), LABELS_FILE), index_col=0)
    labels['pth'] = labels['pth'].apply(lambda x:os.path.join(DATA_DIRECTORY, x))
    # For each processed image, we read it in, convert it to grayscale, flatten it, and save it with its label
    images = [(np.array(cv2.cvtColor(cv2.imread(row['pth']), cv2.COLOR_BGR2GRAY)).flatten(), row['label']) for idx, row in labels.iterrows()]

    # Split X (image array) and Y (label)
    X = [image[0] for image in images]
    Y = [image[1] for image in images]

    # Create test, train datasets
    return train_test_split(X, Y, test_size=TEST_PERCENT)

def save_model(model):
    # Save our model as a pickle
    print("Saving model...")
    knnPickleFile = open(MODEL_SAVE_DIRECTORY, 'wb')

    pickle.dump(model, knnPickleFile)

    knnPickleFile.close()

if __name__ == "__main__":
    print("Reading images and creating training data...")
    x_train, x_test, y_train, y_test = create_datasets()

    print("  -> Performing PCA...")
    pca = PCA(n_components=1000)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    print("Training the model...")
    svc = LinearSVC(verbose=True, max_iter=10000, dual=False)
    # svc = SGDClassifier(verbose=1, n_jobs=-1)
    svc.fit(x_train, y_train)

    save_model(svc)

    print("Testing the model...")
    y2 = svc.predict(x_test)

    print("Accuracy on unknown data is", accuracy_score(y_test, y2))
    print("Accuracy on unknown data is", classification_report(y_test, y2))

    result = pd.DataFrame({'original': y_test, 'predicted': y2})
    print(result)