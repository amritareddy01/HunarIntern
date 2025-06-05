import os
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

base_dir = '/Users/amrita/Desktop/hunarintern/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')


train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
 height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
 batch_size=32, class_mode='binary')

validation_generator = val_datagen.flow_from_directory(validation_dir, target_size=(150, 150),
 batch_size=32, class_mode='binary')

model = tf.keras.models.Sequential([
 tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
 tf.keras.layers.MaxPooling2D(2, 2),
 tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
 tf.keras.layers.MaxPooling2D(2, 2),
 tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
 tf.keras.layers.MaxPooling2D(2, 2),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(512, activation='relu'),
 tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_generator, steps_per_epoch=100, epochs=10,
 validation_data=validation_generator, validation_steps=50)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.tight_layout()
plt.show()

img_path = 'cat.jpg'
img = load_img(img_path, target_size=(150, 150))
x = img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)

if prediction[0] > 0.5:
 print("It's a dog!")
else:
 print("It's a cat!")
