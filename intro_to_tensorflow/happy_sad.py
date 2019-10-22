import tensorflow as tf
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

DESIRED_ACCURACY = 0.999

# !wget --no-check-certificate \
#     "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
#     -O "/tmp/happy-or-sad.zip"

zip_ref = zipfile.ZipFile("happy-or-sad.zip", 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>DESIRED_ACCURACY):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        "/tmp/h-or-s",
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')

# Expected output: 'Found 80 images belonging to 2 classes'

history = model.fit_generator(
    train_generator,
    steps_per_epoch=2,
    epochs=15,
    verbose=1,
    callbacks=[callbacks]
)
###########################

import numpy as np
import cv2

fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    # img = mpimg.imread('/tmp/h-or-s/happy/happy2-03.png')
    img_dir = "/tmp/h-or-s/happy/"
    img_path = img_dir + random.choice(os.listdir("/tmp/h-or-s/happy/"))
    np_img = cv2.imread(img_path)
    img = mpimg.imread(img_path)
    print type(img)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    # print model.predict([img])
plt.show()



# mnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# print type(training_images)
# import io
# with zipfile.ZipFile('happy-or-sad.zip') as zipper:
#     for p in zipper.namelist():
#         with io.BufferedReader(zipper.open(p, mode='r')) as f:
#             print p
#             load = np.load(f)
#             print type(load)
