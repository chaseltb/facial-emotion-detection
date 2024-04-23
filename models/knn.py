import sklearn
import sklearn.model_selection
import sklearn.neighbors
import cv2
import os
import pandas as pd
import numpy as np
import pickle

# Constants
DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "../data")
MODEL_SAVE_DIRECTORY = os.path.join(os.path.dirname(__file__), "../models/trained_models/knn")
LABELS_FILE = "..\\data\\labels.csv"
TEST_PERCENT = 0.2

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
    return sklearn.model_selection.train_test_split(X, Y, test_size = TEST_PERCENT)

def validate(x_train, y_train, min_k=137, max_k=187):
    accuracies = dict()
    # Testing a little below k=sqrt(len(dataset)) ~~ 167
    # 137, 139, 141 .. 187
    for k in range(min_k, max_k+1, 2):
        neighbors = k
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=neighbors)
        # K-folds cross validation to score
        scores = sklearn.model_selection.cross_val_score(knn, x_train, y_train, cv=5)

        # Mean accuracy of k-folds cv is just sum/len of scores array
        mean_accuracy = sum(scores)/len(scores)
        print("k: {}, Mean accuracy {}".format(k, mean_accuracy))
        accuracies[k] = mean_accuracy

    # Sort accuracies by accuracy, counting down
    sorted_accuracies = dict(sorted(accuracies.items(), key= lambda x:x[1], reverse=True))
    print(sorted_accuracies)
    # This gets the first item in the dict (as python dicts are technically ordered)
    best_k = next(iter(sorted_accuracies))
    print("BEST K: {} ACCURACY: {}".format(best_k, sorted_accuracies[best_k]))
    return best_k

def train_and_score(x_train, y_train, x_test, y_test, k=153):
    # Train with our selected K from validation
    print("Training/testing with best K...")

    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    test_accuracy = knn.score(x_test, y_test)
    
    # In testing, k value of 153 returns accuracy of ~0.26
    print("K value: {} Test accuracy: {}".format(best_k, test_accuracy))
    return knn

def confusion_matrix(knn, x_test, y_test):
    # For labels
    classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    # We need a mapping of idx:class for pandas
    mapping = dict()
    for idx, cl in enumerate(classes):
        mapping[idx] = cl

    # Predict y given x_test
    predicted_y = knn.predict(x_test)
    # By saying labels=classes, we ensure the order for labeling
    matrix = sklearn.metrics.confusion_matrix(y_test, predicted_y, labels=classes)

    # Convert to DataFrame for labeling
    df = pd.DataFrame(matrix)
    df.rename(columns=mapping, inplace=True)
    df.rename(index=mapping, inplace=True)
    df.rename_axis("True value", axis="index", inplace=True)
    df.rename_axis("Predicted value ->", axis="columns", inplace=True)
    print(df)

def save_model(knn):
    # Save our model as a pickle
    print("Saving model...")
    knnPickleFile = open(MODEL_SAVE_DIRECTORY, 'wb')

    pickle.dump(knn, knnPickleFile)

    knnPickleFile.close()

if __name__ == "__main__":
    # Load in the data 
    x_train, x_test, y_train, y_test = create_datasets()
    
    # Find k via validation
    # Don't want to do ~30 min of validation? We've determined the best k to be 153
    best_k = 153
    # best_k = validate(x_train, y_train)

    # Train and score our model
    trained_model = train_and_score(x_train, y_train, x_test, y_test, k=best_k)

    # View a confusion matrix of test values
    confusion_matrix(trained_model, x_test, y_test)

    # Save our trained model
    save_model(trained_model)