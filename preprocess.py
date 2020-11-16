import os
import librosa
import math
import json

DATASET_PATH = "Data_reduced"
JSON_PATH = "data.json"

SAMPLE_RATE = 22050
NUM_SAMPLES = SAMPLE_RATE * 3   # each file is 3 seconds long

def create_and_save_mfcc(dataset_path, json_path, n_mfcc, n_fft, hop_length, num_segments):
    """
    Creates and saves MFCCs in a json file

    :param data_path (string): path to dataset
    :param json_path (string): path to new json file
    :param n_mfcc (int): number of mfccs
    :param n_ftt (int): number of ftts
    :param hop_length (int): shift
    :param num_segments (int): number of segments
    :return:
    """

    # store data in dictionary
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(NUM_SAMPLES / num_segments)
    num_mfcc_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # loop through all files
    for (path, folders, files) in os.walk(dataset_path):

        if path != dataset_path and len(folders) == 0:

            for f in files:

                # don't read hidden files (os reads them)
                if f.startswith("."):
                    continue

                fp_componenets = f.split("/")  # ["Data", "Actor_01", "03-01-01-01-01-01-01.wav"]

                filename_ids = fp_componenets[-1].split("-")  # ["03", "01", "01", "01", "01", "01", "01.wav"]
                emotion = filename_ids[2]
                if emotion not in data["mapping"]:
                    data["mapping"].append(emotion)
                index = data["mapping"].index(emotion)

                # load the audio file
                file_path = os.path.join(path, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments
                for s in range(num_segments):

                    # segment
                    start_sample = num_samples_per_segment * s
                    end_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:end_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # store mfcc segment
                    if len(mfcc) == num_mfcc_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(index)  # emotion = "01" -> index 0

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    create_and_save_mfcc(DATASET_PATH, JSON_PATH,
                         n_mfcc=13,
                         n_fft=2048,
                         hop_length=512,
                         num_segments=10)



