import numpy as np
import os
import random
import shutil
from ultralytics import YOLO

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
    f.write("   0: straight ahead\n   1: no U-turn\n   2: no overtaking\n   3: parking\n   4: yield\n   5: tramway\n   6: pedestrian crossing\n   7: no entry\n   8: turn right\n   9: priority\n   10: stop\n   11: no stopping\n   12: no right turns\n   13: no left turns\n   14: no movement in both directions\n   15: roundabout\n   16: dead end\n   17: pedestrian-cyclist crossing\n   18: speed bump\n   19: 30 speed limit\n   20: 60 speed limit\n   21: 50 speed limit\n   22: gas station\n   23: 20 speed limit\n   24: intersection-entry\n   25: end of priority\n   26: turn left or right\n   27: entry to no through road\n   28: keep right\n   29: children crossing\n   30: no entry for trucks\n   31: bus stop\n   32: mechanic services\n   33: 70 speed limit\n   34: pedestrian crossing\n   35: end of speed limit\n   36: children\n   37: bicycle and pedestrian path\n   38: danger\n   39: cyclist\n   40: weight limit\n   41: 40 speed limit\n   42: left turn mandatory\n   43: no entry for\n   44: end of parking\n   45: narrowing\n   46: danger left turn\n   47: traffic lights\n   48: 80 speed limit\n   49: roundabout\n   50: 100 speed limit\n   51: 90 speed limit\n   52: two way traffic\n   53: 10 speed limit\n   54: height limit\n   55: right-left turn\n   56: left-right turn\n   57: right turn\n   58: no parking\n   59: 15 speed limit\n   60: 5 speed limit\n")

model = YOLO("yolov8s.pt")

# train the model
results = model.train(data=os.path.join(prepared_dataset_dir, "detection/data.yaml"), save_dir="./", name="detector_models", epochs=15, imgsz=640, exist_ok=True)