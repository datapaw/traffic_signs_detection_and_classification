import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")

# Define the paths to the dataset
image_dir = "C:\\Users\\Omen\\OneDrive\\Desktop\\upload\\traffic_signs_detection_and_classification\\src\\dataset\\detection\\images"
label_dir = "C:\\Users\\Omen\\OneDrive\\Desktop\\upload\\traffic_signs_detection_and_classification\\src\\dataset\\detection\\labels"

class _signs_dataset():
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = os.listdir(image_dir)
        self.label_files = os.listdir(label_dir)
    
    
        
        
def main():
    data = os.listdir(label_dir)
    print(data)
    
if __name__ == "__main__":
    main()