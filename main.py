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


def get_four_images():
    batch = data_iterator.next()  # len(2) -> pos 0 has images, pos 1 has label
    # Check which image is assigned to which label for first four images
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img)
        ax[idx].title.set_text(batch[1][idx])


get_four_images()

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

train_size = int(len(data) * 0.6)  # train data
val_size = int(len(data) * 0.2) + 1  # evaludate model while training
test_size = int(len(data) * 0.2) + 1  # test after model has trained

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# --------------------------------------------------------------
# Deep learning pipeline
# --------------------------------------------------------------

# one data input and output | functional api is good for multiple connections
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
# (32, 256, 256, 3) -> so image is 256, 256, 3
model.add(Conv2D(16, (3, 3), 1, activation="relu", input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation="relu"))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))  # Good for binary classification

model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])

model.summary()

# --------------------------------------------------------------
# Train -> put in logs
# --------------------------------------------------------------
logdir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# train
hist = model.fit(
    train, epochs=20, validation_data=val, callbacks=[tensorboard_callback]
)

# hist has train acc, val acc, loss, etc
hist.history

# --------------------------------------------------------------
# Plot performance
# --------------------------------------------------------------


def get_loss():
    # val going up while loss going down indicates overfitting
    fig = plt.figure()
    plt.plot(hist.history["loss"], color="teal", label="loss")
    plt.plot(hist.history["val_loss"], color="orange", label="val_loss")
    fig.suptitle("Loss", fontsize=20)
    plt.legend(loc="upper left")
    plt.show()


get_loss()


def get_accuracy():
    fig = plt.figure()
    plt.plot(hist.history["accuracy"], color="teal", label="loss")
    plt.plot(hist.history["val_accuracy"], color="orange", label="val_accuracy")
    fig.suptitle("Accuracy", fontsize=20)
    plt.legend(loc="upper left")
    plt.show()


get_accuracy()

# --------------------------------------------------------------
# Evaluate performance
# --------------------------------------------------------------

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

len(test)


def test_on_test_set():
    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)


test_on_test_set()
print(
    f"Precision {pre.result().numpy()}, Recall {re.result().numpy()}, Accuracy {acc.result().numpy()}"
)

# --------------------------------------------------------------
# Test on data outside of dataset
# --------------------------------------------------------------

# Have to grab, didn't
img = cv2.imread("happytest.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# resize
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()

# model expects batches so need to add extra dimension to image before passing it in
resized_image = np.expand_dims(resize / 255, axis=0)
yhat = model.predict(resized_image)

# above .5 is sad, below is happy

# --------------------------------------------------------------
# Save model
# --------------------------------------------------------------

from tensorflow.keras.models import load_model

# Can use .h5 instead
model.save(os.path.join("models", "image_classifier.keras"))

# --------------------------------------------------------------
# reload model
# --------------------------------------------------------------
new_model = load_model(os.path.join("models", "image_classifier.keras"))
