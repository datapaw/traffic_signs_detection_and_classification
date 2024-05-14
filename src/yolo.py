import numpy as np
import os
import random
import shutil
from ultralytics import YOLO

orig_dataset_dir = "../input/polish-traffic-signs-dataset"
prepared_dataset_dir = "./datasets/"

detection_images_dir = "C:\\Users\\Omen\\OneDrive\\Desktop\\upload\\traffic_signs_detection_and_classification\\src\\dataset\\detection\\images"
detection_labels_dir = "C:\\Users\\Omen\\OneDrive\\Desktop\\upload\\traffic_signs_detection_and_classification\\src\\dataset\\detection\\labels"

detection_train_dir = os.path.join(prepared_dataset_dir, "detection/train")
detection_val_dir = os.path.join(prepared_dataset_dir, "detection/val")
detection_test_dir = os.path.join(prepared_dataset_dir, "detection/test")

# shuffle the data
data = os.listdir(detection_images_dir)
random.shuffle(data)

# 70-10-20 data split
train_data = data[:7 * len(data) // 10]
val_data = data[7 * len(data) // 10: 8 * len(data) // 10]
test_data = data[8 * len(data) // 10:]


def copy_data(data, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # assuming the label file name is the same as image file name
    for image_file_name in data:
        image_path = os.path.join(detection_images_dir, image_file_name)
        label_path = os.path.join(detection_labels_dir, os.path.splitext(image_file_name)[0] + ".txt")
        
        # copy the data into a new folder
        shutil.copy(image_path, target_dir)
        shutil.copy(label_path, target_dir)


# split data into different folders
copy_data(train_data, detection_train_dir)
copy_data(val_data, detection_val_dir)
copy_data(test_data, detection_test_dir)

# prepare the data.yaml file for training the YOLO model
with open(os.path.join(prepared_dataset_dir, "detection/data.yaml"), "w+") as f:
    f.write("path: detection\r\n")
    f.write("train: train\r\n")
    f.write("val: val\r\n")
    f.write("test: test\r\n")
    f.write("names: \r\n")
    f.write("  0: sign")

model = YOLO("yolov8n.pt")

# train the model
results = model.train(data=os.path.join(prepared_dataset_dir, "detection/data.yaml"), save_dir="./", name="detector_models", epochs=15, imgsz=640, exist_ok=True)

