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

if __name__ == "__main__":

    # create train, validation, and test sets
    # build CNN
    # compile CNN
    # train CNN
    # evaluate CNN on test set
    # make prediction on a sample

    predict(model, X, y)