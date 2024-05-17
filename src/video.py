import cv2
import ultralytics

# Load the YOLOv8 model
model = ultralytics.YOLO("./signs_det_clas_results\weights\last.pt")  # Specify the path to your YOLOv8 weights

# Open the video file
video_path = "./video/wawa_HD.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video writer initialized to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./video/output_video_HD.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv8 inference on the frame
    results = model(frame)
    
    # Assuming results is a list of detected objects, we need to plot the detections manually
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confidences = result.boxes.conf  # Confidence scores
        classes = result.boxes.cls  # Class IDs
        
        # Draw the bounding boxes on the frame
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = confidences[i]
            class_id = int(classes[i])
            
            label = f'{model.names[class_id]} {confidence:.2f}'
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Write the frame with detections to the output video
    out.write(frame)
    
    # Display the frame with detections (optional)
    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()