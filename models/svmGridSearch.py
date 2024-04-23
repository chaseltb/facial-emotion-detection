import pandas as pd
import os
from skimage.io import imread
import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle


Categories = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
flat_data_arr = []  # input array
target_arr = []  # output array
datadir = r"..\processed_data"  # Using this b/c resizing when loading takes forever

if not os.path.exists("SVM IMage Data.pkl"):
    for i in Categories:
        print(f'loading... category : {i}')
        path = os.path.join(datadir, i)

        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            # img_resized = resize(img_array, (150, 150, 3))
            flat_data_arr.append(img_array.flatten())
            target_arr.append(Categories.index(i))
        print(f'loaded category:{i} successfully')

        saveFile = open("SVM IMage Data.pkl", "wb")
        pickle.dump([flat_data_arr, target_arr], saveFile)
        saveFile.close()
else:
    print("Loading from pickle file...")
    loadFile = open("SVM IMage Data.pkl", "rb")
    flat_data_arr, target_arr = pickle.load(loadFile)
    loadFile.close()

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

# dataframe
df = pd.DataFrame(flat_data)
df['Target'] = target

# input data
x = df.iloc[:, :-1]
# output data
y = df.iloc[:, -1]
print("Data loaded successfully...")

print("Splitting the data into training and testing sets...")
# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)

print("Creating the model...")
# Defining the parameters grid for GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.0001, 0.001, 0.1, 1],
              'kernel': ['rbf', 'poly']}

# Creating a support vector classifier
svc = svm.SVC(probability=False, verbose=True)

# Creating a model using GridSearchCV with the parameters grid
model = GridSearchCV(svc, param_grid, n_jobs=-1, verbose=True)

print("Training the model...")
# Training the model using the training data
model.fit(x_train, y_train)

print("Testing the model...")
# Testing the model using the testing data
y_pred = model.predict(x_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_pred, y_test)

# Print the accuracy of the model
print(f"The model is {accuracy * 100}% accurate")

print(classification_report(y_test, y_pred, target_names=Categories))
