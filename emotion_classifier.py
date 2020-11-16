import json
import numpy as np

def load_data(dataset_path):

    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # converting lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

if __name__ == "__main__":

    # load data

    # get train and test sets

    # build network

    # compile network

    # train network