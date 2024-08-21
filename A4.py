import glob
import os
from preprocessing import image_dataset_from_directory


files = glob.glob("Monkey Species Data/*/*/*")
for file in files:
    f = open(file, "rb")  # open to read binary file
    if not b"JFIF" in f.peek(10):
        f.close()
        os.remove(file)
    else:
        f.close()


training_set = image_dataset_from_directory("Monkey Species Data/Training Data", label_mode="categorical", image_size=(100,100))

test_set = image_dataset_from_directory("Monkey Species Data/Prediction Data", label_mode="categorical", image_size=(100,100), shuffle=False)
