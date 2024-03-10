import tensorflow as tf
import os

# --------------------------------------------------------------
# List potential gpus
# --------------------------------------------------------------

gpus = tf.config.experimental.list_physical_devices("GPU")

# --------------------------------------------------------------
# Remove dodgy images
# --------------------------------------------------------------

import cv2
import imghdr
from matplotlib import pyplot as plt

data_dir = "data"
image_exts = ["jpeg", "jpg", "bmp", "png"]

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)

        try:
            img = cv2.imread(image_path)
            extension = imghdr.what(image_path)
            if extension not in image_exts:
                print(f"Image not in ext list {image_path}")
                os.remove(image_path)
        except:
            print(f"Issue with image {image_path}")
# example without color correction
img = cv2.imread(os.path.join(data_dir, "happy", "_happy_jumping_on_beach-40815.jpg"))
plt.imshow(img)

# with color correction
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt

data = tf.keras.utils.image_dataset_from_directory("data")
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()  # len(2) -> pos 0 has images, pos 1 has label

# Check which image is assigned to which label for first four images
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])

# --------------------------------------------------------------
# Preprocessing images
# --------------------------------------------------------------

# image shape
batch[0].shape

# normalize
data = data.map(lambda x, y: (x / 255, y))
data.as_numpy_iterator().next()[0]

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

train_size = int(len(data) * 0.7)  # train data
val_size = int(len(data) * 0.2)  # evaludate model while training
test_size = int(len(data) * 0.1)  # test after model has trained

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

