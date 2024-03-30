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
print(myResNet.summary())

# freeze the weights
for layer in myResNet.layers:
    layer.trainable = False

Classes = glob(trainMyImagesFolder + "/*")
print(Classes)

numOfClasses = len(Classes)

global_average_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(myResNet.output)

PlusFlatten = Flatten()(global_average_pooling_layer)

predictionLayer = Dense(numOfClasses, activation="softmax")(PlusFlatten)

model = Model(inputs=myResNet.input, outputs=predictionLayer)
print(model.summary())

model.compile(
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)

# data augmentation

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
training_set = train_datagen.flow_from_directory(
    trainMyImagesFolder, target_size=(224, 224), batch_size=32, class_mode="categorical"
)



test_set = test_datagen.flow_from_directory(
    testMyImagesFolder, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

EPOCHS = 50
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
plt.text(0.5, 0.5, f"Training time: {int(hours)}:{int(minutes)}:{int(seconds)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.legend()
plt.show()

