import pickle
import os

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import metrics

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

def confusion_matrix(y_test, y2):
    # For labels
    mapping = {i: CATEGORIES[i] for i in range(8)}

    # By saying labels=classes, we ensure the order for labeling
    matrix = metrics.confusion_matrix(y_test, y2, labels=CATEGORIES)

    # Convert to DataFrame for labeling
    df = pd.DataFrame(matrix)
    df.rename(columns=mapping, inplace=True)
    df.rename(index=mapping, inplace=True)
    df.rename_axis("True value", axis="index", inplace=True)
    df.rename_axis("Predicted value ->", axis="columns", inplace=True)
    print(df.to_string())

def train():
    print("Reading images and creating training data...")
    x_train, x_test, y_train, y_test = create_datasets()

    print("  -> Performing PCA...")
    pca = PCA(n_components=1000)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    print("Training the model...")
    svc = LinearSVC(verbose=True, max_iter=10000, dual=False)
    svc.fit(x_train, y_train)

    save_model(svc)

    print("Testing the model...")
    y2 = svc.predict(x_test)

    print(f"Testing accuracy: {round(accuracy_score(y_test, y2), 3) * 100}%")

    # By saying labels=classes, we ensure the order for labeling
    confusion_matrix(y_test, y2)

if __name__ == "__main__":
    train()
