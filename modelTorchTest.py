import os
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
import torch.nn as nn

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the saved model state dictionary
model_state_dict = torch.load("./temp/model_state_regular.pth")

train_data_path = "./alien_vs_predator_thumbnails/data/train"
train_dataset = datasets.ImageFolder(train_data_path, transform=transform)


# Create the model architecture
model = models.resnet50(pretrained=True)  # Assuming you used ResNet50 architecture
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))


# Load the model state dictionary
model.load_state_dict(model_state_dict)

# Set the model to evaluation mode
model.eval()

# Define the class labels
classes = os.listdir("./alien_vs_predator_thumbnails/data/train")

# Function to prepare the image for prediction
def prepare_image(file):
    img = Image.open(file)
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

# Test images
test_image1 = "./alien_vs_predator_thumbnails/data/validation/alien/16.jpg"
test_image2 = "./alien_vs_predator_thumbnails/data/validation/predator/10.jpg"

# Prepare images for prediction
img1 = prepare_image(test_image1)
img2 = prepare_image(test_image2)

# Perform prediction
with torch.no_grad():
    output1 = model(img1)
    output2 = model(img2)

# Get predicted classes
pred_class1 = torch.argmax(F.softmax(output1, dim=1), dim=1).item()
pred_class2 = torch.argmax(F.softmax(output2, dim=1), dim=1).item()

# Get class names
class_name1 = classes[pred_class1]
class_name2 = classes[pred_class2]

print("Predicted class for Image 1:", class_name1)
print("Predicted class for Image 2:", class_name2)
