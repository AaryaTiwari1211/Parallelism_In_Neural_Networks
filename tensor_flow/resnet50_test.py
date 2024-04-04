import tensorflow as tf
import cv2 as cv
import os
from keras.utils import load_img , img_to_array 
import numpy as np


IMAGE_SIZE = 224
trainImagesFolder = "./alien_vs_predator_thumbnails/data/train"
CLASSES = os.listdir(trainImagesFolder)
num_classes = len(CLASSES)

best_model_file = "./temp/best_model.keras"
model = tf.keras.models.load_model(best_model_file)

def prepare_image(file):
    img = load_img(file, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create batch axis
    img_result = img_array / 255.0
    return img_result

testImage1 = "./alien_vs_predator_thumbnails/data/validation/alien/16.jpg"
testImage2 = "./alien_vs_predator_thumbnails/data/validation/420.jpeg"

img1 = cv.imread(testImage1)
img2 = cv.imread(testImage2)

imgForModel1 = prepare_image(testImage1)
imgForModel2 = prepare_image(testImage2)

result1 = model.predict(imgForModel1, verbose=1)
result2 = model.predict(imgForModel2, verbose=1)
print(result1)
print(result2)

answer1 = np.argmax(result1 , axis=1)
print(answer1)

answer2 = np.argmax(result2 , axis=1)
print(answer2)

index1 = answer1[0]
index2 = answer2[0]

className1 = CLASSES[index1]
className2 = CLASSES[index2]

print("Predicted class for Image 1: " + className1)
print("Predicted class for Image 2: " + className2)
