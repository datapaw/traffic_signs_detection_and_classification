from ultralytics import YOLO
import cv2

model = YOLO("det_clas.pt")

results = model.predict(source="0", show=True)

# model.conf = 0.01 # confidence threshold (0-1)
# model.iou = 0.01  # NMS IoU threshold (0-1)

print(results)