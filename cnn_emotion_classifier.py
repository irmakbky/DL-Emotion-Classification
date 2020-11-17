import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = ""

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y

def train_val_test_split(test_size, validation_size):

    # load data
    # create train/test split
    # create train/validation split

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):

    # create model

    # 1st conv layer
    # 2nd conv layer
    # 3rd conv layer
    # output layer

    return model

def predict(model, X, y):

    X = X[np.newaxis, ...]

    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))

if __name__ == "__main__":

    # create train, validation, and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = train_val_test_split(0.25, 0.2)

    # build CNN
    model = build_model((X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    # compile CNN
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # train CNN
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation),
              batch_size=32,
              epochs=30)

    # evaluate CNN on test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # make prediction on a sample
    X = X_test[100]
    y = y_test[100]

    predict(model, X, y)