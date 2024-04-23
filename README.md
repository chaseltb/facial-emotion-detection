# facial-emotion-detection

![Facial Emotion Detection](./labeled_faces.png)

## Installing Required Libraries
Install the required libraries for this can be installed via

`pip install -r ./requirements.txt`

Please note that pytorch can be installed via the [pytorch website](https://pytorch.org/get-started/locally/) to allow training via GPU.
## Dataset

The dataset in its full form is found in ./data. The dataset can also be found and downloaded online [here](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data/data).

## Training Models

There are three possible models to train in the dataset (SVM, KNN, and CNN). These are found in the ./models file as Python files. To train these models, run their corresponding files in ./models. These files will split the dataset into a train and test set, train the model, save the model in ./models/trained_models, and test the model. The test accuracy for the model, as well as the confusion matrix for the model, will be outputted once training is complete.

Please note that for the KNN, validation has shown that k=153 is the best possible k value. ./models/knn.py contains the code to validate the value of K, but, by default, it only trains a model at k=153. Refer to lines 101-102 to enable validation, but know that this will increase training time to 20-40 minutes.

## Viewing the Live Feed

Running ./live_feed.py will attempt to create a connection to the camera, identify faces in frame, and will assign an emotion to the face. Live_feed utilizes the most accurate model, best-cnn.