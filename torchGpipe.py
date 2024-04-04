import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import time
import os
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time
from torch.cuda.amp import autocast, GradScaler


# Define epochs
max_epochs = 50

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define dataset paths
train_data_path = "./alien_vs_predator_thumbnails/data/train"
test_data_path = "./alien_vs_predator_thumbnails/data/validation"

# Load datasets
train_dataset = datasets.ImageFolder(train_data_path, transform=transform)
test_dataset = datasets.ImageFolder(test_data_path, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Load pre-trained ResNet50 model
myResNet = models.resnet50(pretrained=True)
num_features = myResNet.fc.in_features

# Modify the last fully connected layer to match the number of classes
myResNet.fc = nn.Linear(num_features, len(train_dataset.classes))

# Freeze the pre-trained layers
for param in myResNet.parameters():
    param.requires_grad = False

# Unfreeze the final fully connected layer
for param in myResNet.fc.parameters():
    param.requires_grad = True

# Move model to device
myResNet = myResNet.to(device)

# Wrap the model within nn.Sequential
myResNet = nn.Sequential(myResNet)

# Wrap the model with GPipe
myResNet = GPipe(myResNet, balance=[1], devices=[0, 1, 2, 3], chunks=8)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer (note: GPipe works best with AdamW optimizer)
optimizer = optim.AdamW(myResNet.parameters(), lr=0.01)

# Training loop
start_time = time.time()

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

accumulation_steps = 2

# for epoch in range(max_epochs):
#     myResNet.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     # Gradient accumulation loop 
    
#     # for i, (images, labels) in enumerate(train_loader):
#     #     images, labels = images.to(device), labels.to(device)
        
#     #     optimizer.zero_grad()
        
#     #     outputs = myResNet(images)
#     #     loss = criterion(outputs, labels)
#     #     loss = loss / accumulation_steps # Normalize our loss (if averaged)
#     #     loss.backward()
        
#     #     if (i+1) % accumulation_steps == 0: # Wait for several backward steps
#     #         optimizer.step() # Now we can do an optimizer step
#     #         optimizer.zero_grad() # Reset gradients tensors
        
#     #     running_loss += loss.item()
#     #     _, predicted = outputs.max(1)
#     #     total += labels.size(0)
#     #     correct += predicted.eq(labels).sum().item()
    
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
        
#         optimizer.zero_grad()
        
#         outputs = myResNet(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
    
#     train_loss = running_loss / len(train_loader)
#     train_accuracy = 100. * correct / total
#     train_losses.append(train_loss)
#     train_accuracies.append(train_accuracy)
    
#     # Evaluation on the test set
#     myResNet.eval()
#     test_loss = 0.0
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = myResNet(images)
#             loss = criterion(outputs, labels)
            
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
    
#     test_loss /= len(test_loader)
#     test_accuracy = 100. * correct / total
#     test_losses.append(test_loss)
#     test_accuracies.append(test_accuracy)
    
#     print(f"Epoch [{epoch+1}/{max_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


#Using mixed precision training with GPipe for faster training

scaler = GradScaler()

for epoch in range(max_epochs):
    myResNet.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Casts operations to mixed precision
        with autocast():
            outputs = myResNet(images)
            loss = criterion(outputs, labels)
        
        # Scales loss, and calls backward() to create scaled gradients
        scaler.scale(loss).backward()
        
        # Unscales gradients and calls optimizer.step()
        scaler.step(optimizer)
        
        # Updates the scale for next iteration
        scaler.update()
        
        optimizer.zero_grad()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # Evaluation on the test set
    myResNet.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = myResNet(images)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    print(f"Epoch [{epoch+1}/{max_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Training time: {int(hours)}:{int(minutes)}:{int(seconds)}")

# Save the model's state dictionary
torch.save(myResNet.state_dict(), 'temp/model_state_gpipe_large_batch.pth')

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()

# Add elapsed time to the graph
elapsed_time_str = f"Total training time: {int(hours)}:{int(minutes)}:{int(seconds)}"
plt.annotate(elapsed_time_str, xy=(0.5, 0.5), xycoords='axes fraction', fontsize=12,
             ha='center', va='center')

plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Test Accuracy")
plt.legend()

# Add elapsed time to the graph
plt.annotate(elapsed_time_str, xy=(0.5, 0.5), xycoords='axes fraction', fontsize=12,
             ha='center', va='center')

plt.show()
