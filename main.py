from ultralytics import YOLO
import cv2
import numpy as np

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  
        colorsBGR = [x, y]
        print(colorsBGR)


# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' or other variants as needed.

# Video input and output paths
video_path = 'input_video2.mp4'
output_path = 'output_video2.mp4'

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

coords = [[335, 620], [377, 620], [524, 992], [185, 992]] # bus video
# coords = [[197, 318], [390, 318], [174, 35], [150, 35]] # traffic cam video

bus_lane_polygon = np.array([[coords[0][0], coords[0][1]], 
                                [coords[1][0], coords[1][1]], 
                                [coords[2][0], coords[2][1]], 
                                [coords[3][0], coords[3][1]]], dtype=np.int32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the bus lane region
    cv2.polylines(frame, [bus_lane_polygon], isClosed=True, color=(255, 0, 0), thickness=2)

    # Run YOLO inference
    results = model(frame)

    # Process detections
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)  # Class ID
            conf = box.conf  # Confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Class IDs for vehicles
            class_names = model.names
            is_bus = cls == 5  # COCO class ID for "bus"
            is_vehicle = cls in [2, 3, 5, 7]  # Car, motorcycle, bus, truck

            if is_vehicle:
                # Check if the vehicle is in the bus lane
                if cv2.pointPolygonTest(bus_lane_polygon, bbox_center, measureDist=False) >= 0 and not is_bus:
                    box_color = (0, 0, 255)  # Red for violations
                else:
                    box_color = (0, 255, 0)  # Green for normal vehicles

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Add label
                label = f"{class_names[cls]} {float(conf):.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('YOLOv8 Bus Lane Detection', frame)
    cv2.setMouseCallback('YOLOv8 Bus Lane Detection', POINTS)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
