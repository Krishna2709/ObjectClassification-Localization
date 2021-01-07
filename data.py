# Preparing the dataset

# 1. Import required modules
import glob
import os
from itertools import count
from collections import defaultdict, namedtuple
import cv2
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

# 2. Downloading and parsing the dataset
DATASET_DIR = "dataset"

for type in ("annotations", "images"):
    tf.keras.utils.get_file(
        type, 
        f"https://www.robots.ox.ac.uk/~vgg/data/pets/data/{type}.tar.gz",
        untar=True,
        cache_dir=".",
        cache_subdir=DATASET_DIR
    )

# 3. Locating the data
IMAGE_SIZE = 224
IMAGE_ROOT = os.path.join(DATASET_DIR, "images")
XML_ROOT = os.path.join(DATASET_DIR, "annotations")

# required data from the XML file
Data = namedtuple("Data", "image, box, size, type, breed")

# 4. dictionaries to store breeds and types as numbers
types = defaultdict(count().__next__)
breeds = defaultdict(count().__next__)

# 5. Helper funtion to give path to an XML file, returns an instance of data
def parse_xml(path:str) -> Data:
    # 1. open xml file and parse it
    with open(path) as f:
        xml_string = f.read()
    root = ET.fromstring(xml_string)
    
    # 2. get name of corresponding image and breed
    img_name = root.find("./filename").text
    breed_name = img_name[:img_name.rindex("_")]
    breed_id = breeds[breed_name]

    # 3. Get ID of types
    type_id = types[root.find("./object/name").text]

    # 4. Extract the bounding box and normalize 
    box = np.array([int(root.find(f"./object/bndbox/{tag}").text)
                    for tag in "xmin,ymin,xmax,ymax".split(",")])
    size = np.array([int(root.find(f"./size/{tag}").text)
                    for tag in "width,height".split(",")])
    norm_box = (box.reshape((2,2))/size).reshape((4))
    
    # 5. Return the results as an instance of Data
    return Data(img_name, norm_box, size, type_id, breed_id)

# 6. Parse the dataset
xml_paths = glob.glob(os.path.join(XML_ROOT, "xmls", "*.xml"))
xml_paths.sort()
parsed = np.array([parse_xml(path) for path in xml_paths])

# Available breeds and types
print(f"{len(types)} TYPES:", *types.keys(), sep=", ")
print(f"{len(breeds)} BREEDS:", *breeds.keys(), sep=", ")

# shuffling the datset
np.random.seed(1)
np.random.shuffle(parsed)

# Creating a TensorFlow dataset

# transforming the parsed array
ds = tuple(np.array(list(i)) for i in np.transpose(parsed))
ds_slices = tf.data.Dataset.from_tensor_slices(ds)

# looking at one element from the dataset
for el in ds_slices.take(1):
    print("Single element from TF dataset",el)

# check whether all bounding boxes are correct
for el in ds_slices:
    b = el[1].numpy()
    if(np.any((b>1) | (b<0)) or np.any(b[2:]-b[:2] < 0)):
        print(f"Invalid box found {b} image: {el[0].numpy()}")

# Helper function to transfer the data element to feed into NN
def prepare(image, box, size, type, breed):
    image = tf.io.read_file(IMAGE_ROOT+"/"+image)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image,(IMAGE_SIZE, IMAGE_SIZE))
    image /= 255
    return Data(image, box, size, tf.one_hot(type, len(types)), tf.one_hot(breed, len(breeds)))

# mapping the dataset with the function
ds = ds_slices.map(prepare).prefetch(32)

# Illustrating some samples of the data

if __name__ == "__main__":
    def illustrate(sample):
        breed_num = np.argmax(sample.breed)
        for breed, num in breeds.items():
            if num == breed_num:
                break
        image = sample.image.numpy()
        pt1, pt2 = (sample.box.numpy().reshape((2,2))*IMAGE_SIZE).astype(np.int32)
        cv2.rectangle(image, tuple(pt1), tuple(pt2), (0, 1, 0))
        cv2.putText(image, breed, (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 1, 0))
        return image

    sample_images = np.concatenate([illustrate(sample)
                                    for sample in ds.take(3)])
    cv2.imshow("samples", sample_images)
    cv2.waitKey(0)