import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import kagglehub
import pathlib
import os
import sklearn

from tensorflow import keras
from keras._tf_keras.keras import layers, Sequential
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.utils import get_file, image_dataset_from_directory, load_img, img_to_array
from keras._tf_keras.keras.losses import SparseCategoricalCrossentropy
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.optimizers import schedules, Adam
from sklearn.metrics import confusion_matrix, classification_report

path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print("Path to dataset files:", path)


# Specify the path to your local dataset directory
local_data_dir = "/root/.cache/kagglehub/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1"
test_dir = os.path.join(local_data_dir,"Testing")
train_dir = os.path.join(local_data_dir,"Training")

# Convert the local path to a pathlib Path object
test_dir = pathlib.Path(test_dir)

train_dir = pathlib.Path(train_dir)

#loading train and test datasets
batch_size = 32
img_height = 180
img_width = 180

train_ds = image_dataset_from_directory(
  train_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names

test_ds = image_dataset_from_directory(
  test_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


#performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Standarize data
normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

#creating the model
num_classes = 4
data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)



model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])


#compiling model with Adam optimizer
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
model.compile(optimizer=Adam(learning_rate=lr_schedule),
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train the model
epochs=20
class_weights = {
    0: 1.7,  # Increase weight for Class 0
    1: 1.2,  # Slightly increase weight for Class 1
    2: 1.0,  # No change for Class 2
    3: 1.2   # No change for Class 3
}
history = model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs,
  class_weight=class_weights
)


no_tumor_path = "/root/no_tumor.png"
meningioma_path = "/root/meningioma.png"
glioma_path = "/root/glioma.png"
pituitary_path = "/root/pituitary.png"

img = load_img(
    glioma_path, target_size=(img_height, img_width)
)
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
img2 = load_img(
    no_tumor_path, target_size=(img_height, img_width)
)
img_array2 = img_to_array(img2)
img_array2 = tf.expand_dims(img_array2, 0) # Create a batch

predictions = model.predict(img_array2)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
img3 = load_img(
    meningioma_path, target_size=(img_height, img_width)
)
img_array3 = img_to_array(img3)
img_array3 = tf.expand_dims(img_array3, 0) # Create a batch

predictions = model.predict(img_array3)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
img4 = load_img(
    pituitary_path, target_size=(img_height, img_width)
)
img_array4 = img_to_array(img4)
img_array4 = tf.expand_dims(img_array4, 0) # Create a batch

predictions = model.predict(img_array4)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

true_labels = []  # Replace with your actual labels
predicted_labels = []

for images, labels in test_ds:
    preds = model.predict(images)
    preds_classes = np.argmax(preds, axis=1)
    true_labels.extend(labels.numpy())
    predicted_labels.extend(preds_classes)


cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(true_labels, predicted_labels, target_names=class_names))
