import glob
import os
from random import shuffle
from PIL import Image
import shutil

"""
Split the dataset into a train and test set
Concretely move some images and annotations from dataset/train/ to dataset/test/
"""

train_path = "dataset/train/"
test_path = "dataset/test/"
annotations_folder = "annotations/"
images_folder = "images/"

if __name__ == "__main__":
    images_path = glob.glob(train_path + images_folder + "*.jp*g")
    shuffle(images_path)
    image_count = len(images_path)
    images_test_count = int(image_count * 0.1)
    images_path = images_path[:images_test_count]

    print("Moving ", images_test_count, " images for test set")

    if not os.path.exists(test_path + annotations_folder):
        os.makedirs(test_path + annotations_folder)

    if not os.path.exists(test_path + images_folder):
        os.makedirs(test_path + images_folder)

    for image_path in images_path:
        im = Image.open(image_path)

        width, height = im.size

        image_name = os.path.splitext(os.path.basename(image_path))[0]

        annotation_path_source = os.path.join(train_path, annotations_folder, image_name + ".xml")
        annotation_path_dest = os.path.join(test_path, annotations_folder, image_name + ".xml")

        image_path_dest = os.path.join(test_path, images_folder, os.path.basename(image_path))

        shutil.move(annotation_path_source, annotation_path_dest)
        shutil.move(image_path, image_path_dest)

    print("Done")
