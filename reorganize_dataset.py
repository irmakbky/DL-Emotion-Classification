import shutil, os

def reorganize(dataset_path, new_path):
    os.makedirs(new_path)

    for dirpath, dirnames, filenames in os.walk(dataset_path):

        if dirpath != dataset_path and len(dirnames) == 0:

            for f in filenames:

                # don't read hidden files (os reads them)
                if f.startswith("."):
                    continue

                fp_componenets = f.split("/")  # ["Data", "Actor_01", "03-01-01-01-01-01-01.wav"]

                filename_ids = fp_componenets[-1].split("-")  # ["03", "01", "01", "01", "01", "01", "01.wav"]
                emotion = filename_ids[2]
                if not os.path.exists(os.path.join(new_path, emotion)):
                    os.makedirs(os.path.join(new_path, emotion))

                shutil.copy(os.path.join(dirpath, f), os.path.join(new_path, emotion))

if __name__ == "__main__":
    reorganize("smaller_set", "re_smaller")
