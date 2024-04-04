import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import time

# Dataset used - https://www.kaggle.com/datasets/pmigdal/alien-vs-predator-images

IMAGE_SIZE = [224, 224]

trainMyImagesFolder = "./alien_vs_predator_thumbnails/data/train"
testMyImagesFolder = "./alien_vs_predator_thumbnails/data/validation"


myResNet = ResNet50(input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False)

"""
It loads the pre-trained ResNet50 model from TensorFlow's Keras Applications, specifying the input shape 
to match the images in the dataset (224x224 pixels with 3 color channels) and setting include_top=False to 
exclude the final classification layer, allowing for customization 

"""

print(myResNet.summary())

# freeze the weights
for layer in myResNet.layers:
    layer.trainable = False

"""
The weights of the pre-trained ResNet50 model are frozen to prevent them from 
being updated during training. This is a common practice in transfer learning 
to leverage the pre-trained model's learned features without retraining the 
entire model.

"""

Classes = glob(trainMyImagesFolder + "/*")
print(Classes)

numOfClasses = len(Classes)

global_average_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(myResNet.output)

PlusFlatten = Flatten()(global_average_pooling_layer)

predictionLayer = Dense(numOfClasses, activation="softmax")(PlusFlatten)

"""
The code adds custom layers to the model for classification. 
It uses a GlobalAveragePooling2D layer to reduce the spatial dimensions of the feature maps, 
followed by a Flatten layer to flatten the output into a 1D vector. 
Finally, a Dense layer with a softmax activation function is added to perform the classification, 
with the number of units equal to the number of classes in the dataset
"""

model = Model(inputs=myResNet.input, outputs=predictionLayer)
print(model.summary())

model.compile(
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)
"""
The model is compiled with the Adam optimizer, 
categorical cross-entropy loss, and accuracy as the metric. 
The learning rate for the Adam optimizer is set to 0.01.

"""

# data augmentation

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

"""
The code sets up data augmentation for the training set using ImageDataGenerator, 
which includes rescaling, shear, zoom, and horizontal flip transformations. 
This helps to increase the diversity of the training data 
and improve the model's ability to generalize.

"""

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
training_set = train_datagen.flow_from_directory(
    trainMyImagesFolder, target_size=(224, 224), batch_size=32, class_mode="categorical"
)


test_set = test_datagen.flow_from_directory(
    testMyImagesFolder, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

EPOCHS = 12
best_model_file = "./temp/best_model.keras"

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

callbacks = [
    ModelCheckpoint(
        filepath=best_model_file, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=10, min_lr=1e-6),
    EarlyStopping(monitor="val_accuracy", patience=30, verbose=1),
]

start_time = time.time()

r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    callbacks=callbacks,
)

end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)


best_val_acc = max(r.history["val_accuracy"])
print("Best validation accuracy: ", best_val_acc)

"""
The model is trained using the fit method, with the training and validation datasets specified. 
Callbacks are used to save the best model based on validation accuracy, 
reduce the learning rate when the validation accuracy plateaus, 
and stop training early if the validation accuracy does not 
improve for a specified number of epochs.
"""

# plt.plot(r.history["accuracy"], label="train acc")
# plt.plot(r.history["val_accuracy"], label="val acc")
# plt.legend()
# plt.show()

plt.plot(r.history["loss"], label="train loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(r.history["accuracy"], label="train acc")
plt.plot(r.history["val_accuracy"], label="val acc")
plt.text(
    0.5,
    0.5,
    f"Training time: {int(hours)}:{int(minutes)}:{int(seconds)}",
    horizontalalignment="center",
    verticalalignment="center",
    transform=plt.gca().transAxes,
)
plt.legend()
plt.show()

"""
After training, the code calculates the best validation accuracy achieved during training and plots 
the training and validation loss and accuracy over epochs. This helps in understanding the model's 
performance and identifying overfitting or underfitting.

"""
