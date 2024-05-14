import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Define the paths to the dataset
image_dir = "C:\\Users\\Omen\\OneDrive\\Desktop\\upload\\traffic_signs_detection_and_classification\\src\\dataset\\detection\\images"
label_dir = "C:\\Users\\Omen\\OneDrive\\Desktop\\upload\\traffic_signs_detection_and_classification\\src\\dataset\\detection\\labels"

# Define a custom dataset class
class TrafficSignDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.label_files = os.listdir(label_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        label_name = os.path.join(self.label_dir, self.label_files[idx])
        
        # Load image
        try:
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image: {img_name}")
            print(e)
            return None, None
        
        # Load label (you may need to parse the label file)
        with open(label_name, 'r') as f:
            label = f.read().strip()  # Assuming label is stored as text
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define the neural network model
class TrafficSignClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TrafficSignClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define transformations to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to tensor
    # Add more transformations here as needed (e.g., normalization)
])

# Create dataset and data loader
dataset = TrafficSignDataset(image_dir, label_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = TrafficSignClassifier(num_classes=10)  # Assuming there are 10 classes of traffic signs
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
        if images is None:
            # Skip this batch if images are None
            continue
        
        images = images.to(device)
        labels = torch.tensor([int(label) for label in labels])  # Convert labels to tensor
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/100:.4f}")
            running_loss = 0.0

print("Training finished.")
