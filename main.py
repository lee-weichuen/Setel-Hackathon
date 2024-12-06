from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' or other variants as needed.

# Video input and output paths
video_path = 'input_video.mp4'
output_path = 'output_video.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Define the hardcoded bus lane region (adjusted as a percentage of frame height)
bus_lane_start_ratio = 0.5  # Start of the bus lane as a fraction of frame height
bus_lane_end_ratio = 0.6    # End of the bus lane as a fraction of frame height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate bus lane region
    bus_lane_y1 = int(frame_height * bus_lane_start_ratio)
    bus_lane_y2 = int(frame_height * bus_lane_end_ratio)

    # Draw the bus lane region
    cv2.rectangle(frame, (0, bus_lane_y1), (frame_width, bus_lane_y2), (255, 0, 0), 2)  # Blue rectangle

    # Run YOLO inference
    results = model(frame)

    # Process detections
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)  # Class ID
            conf = box.conf  # Confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            # Class IDs for vehicles
            class_names = model.names
            is_bus = cls == 5  # COCO class ID for "bus"
            is_vehicle = cls in [2, 3, 5, 7]  # Car, motorcycle, bus, truck

            if is_vehicle:
                # Check if the vehicle is in the bus lane
                is_in_bus_lane = y2 > bus_lane_y1 and y1 < bus_lane_y2

                # Set bounding box color (red if non-bus vehicle in bus lane)
                if is_in_bus_lane and not is_bus:
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
