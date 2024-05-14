import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import cv2
import os
from torchvision.datasets.folder import default_loader

# Define the neural network architecture
class TrafficSignClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TrafficSignClassifier, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 20, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(20 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom dataset class to load images and labels
class ImageLabelFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        super(ImageLabelFolder, self).__init__(root, transform=transform, target_transform=target_transform, loader=loader)
        self.label_dir = root

    def __getitem__(self, index):
        img_path, _ = self.imgs[index]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
        with open(label_path, 'r') as file:
            label = int(file.readline().strip())  # Assuming labels are integers
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# Define data preprocessing and augmentation
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define the paths to the image and label directories
image_dir = "C:\\Users\\Omen\\OneDrive\\Desktop\\upload\\traffic_signs_detection_and_classification\\src\\dataset\\detection\\images"
label_dir = "C:\\Users\\Omen\\OneDrive\\Desktop\\upload\\traffic_signs_detection_and_classification\\src\\dataset\\detection\\labels"

# Load the dataset
dataset = ImageLabelFolder(root=image_dir, transform=data_transforms)

# Define data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = TrafficSignClassifier(num_classes=len(dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Initialize the camera
cap = cv2.VideoCapture(0)

# Real-time inference loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    frame = cv2.resize(frame, (32, 32))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    frame = frame / 255.0
    frame = (frame - 0.5) / 0.5
    frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(frame)
        _, predicted = torch.max(outputs, 1)

    # Display the result
    cv2.putText(frame, dataset.classes[predicted.item()], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Traffic Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
