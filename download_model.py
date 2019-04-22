import os
import shutil
import tarfile
import urllib.request

"""
Download a pretrained TF detection model
"""


MODEL = "ssdlite_mobilenet_v2_coco_2018_05_09"

MODEL_FILE = MODEL + ".tar.gz"
DOWNLOAD_BASE = "http://download.tensorflow.org/models/object_detection/"
DEST_DIR = "pretrained_model"

if __name__ == "__main__":
    if not (os.path.exists(MODEL_FILE)):
        print("Downloading model...")
        urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

    tar = tarfile.open(MODEL_FILE)
    tar.extractall()
    tar.close()

    os.remove(MODEL_FILE)
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.rename(MODEL, DEST_DIR)
    print("Done.")
