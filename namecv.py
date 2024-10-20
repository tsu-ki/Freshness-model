import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from collections import Counter
from datetime import datetime, timedelta
import threading
import queue

# Load the pre-trained model
model = load_model("Indian_Fruit_98.25.h5")
img_size = (224, 224)

class_names = [
    "Apple_Bad", "Apple_Good", "Apple_mixed",
    "Banana_Bad", "Banana_Good", "Banana_mixed",
    "Guava_Bad", "Guava_Good", "Guava_mixed",
    "Lemon_mixed",
    "Lime_Bad", "Lime_Good",
    "Orange_Bad", "Orange_Good", "Orange_mixed",
    "Pomegranate_Bad", "Pomegranate_Good", "Pomegranate_mixed"
]

# Shelf life information
shelf_life = {
    "Apple": {"shelf": "1-2 days", "refrigerator": "3 weeks", "freezer": "8 months (cooked)"},
    "Banana": {"shelf": "Until ripe", "refrigerator": "2 days (skin will blacken)",
               "freezer": "1 month (whole peeled)"},
    "Guava": {"shelf": "3-5 days", "refrigerator": "1 week", "freezer": "Do not freeze"},
    "Lemon": {"shelf": "10 days", "refrigerator": "1-2 weeks", "freezer": "Do not freeze"},
    "Lime": {"shelf": "10 days", "refrigerator": "1-2 weeks", "freezer": "Do not freeze"},
    "Orange": {"shelf": "10 days", "refrigerator": "1-2 weeks", "freezer": "Do not freeze"},
    "Pomegranate": {"shelf": "1-2 days", "refrigerator": "3-4 days", "freezer": "Balls, 1 month"}
}


def preprocess_for_model(image):
    image = cv2.resize(image, img_size)
    image = preprocess_input(image)
    return np.expand_dims(image, axis=0)


@tf.function
def predict(input_tensor):
    return model(input_tensor, training=False)


def non_max_suppression(boxes, scores, overlapThresh=0.4):
    if len(boxes) == 0:
        return [], []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]
    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[1:]]
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlapThresh)[0] + 1)))
    return boxes[pick].astype("int"), scores[pick]


def calculate_diameter(box):
    return max(box[2] - box[0], box[3] - box[1])


def parse_days(day_string):
    # Handle ranges like "3-4 days" or single values like "3 days"
    parts = day_string.split()
    if '-' in parts[0]:
        low, high = map(int, parts[0].split('-'))
        return (low + high) // 2  # Return average of range
    return int(parts[0])


def get_shelf_life_info(fruit_class):
    fruit_type = fruit_class.split('_')[0]
    condition = fruit_class.split('_')[1]

    if fruit_type in shelf_life:
        info = shelf_life[fruit_type]
        refrigerator_days = parse_days(info["refrigerator"])

        if condition == "Bad":
            shelf_days = min(1, refrigerator_days)
        elif condition == "mixed":
            shelf_days = max(1, int(refrigerator_days * 0.7))
        else:  # "Good"
            shelf_days = refrigerator_days

        return {
            "shelf": info["shelf"],
            "refrigerator": f"{shelf_days} days",
            "freezer": info["freezer"],
            "estimated_days": shelf_days
        }
    else:
        return {
            "shelf": "Unknown",
            "refrigerator": "Unknown",
            "freezer": "Unknown",
            "estimated_days": 3  # Default to 3 days if unknown
        }


def detect_and_predict(frame):
    original_height, original_width = frame.shape[:2]
    scale_factor = min(1, 640 / max(original_height, original_width))  # Limit max dimension to 640px
    if scale_factor < 1:
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    height, width = frame.shape[:2]
    window_size = (224, 224)
    stride = 112  # Reduced stride for better coverage

    detections = []
    scores = []

    for y in range(0, height - window_size[1] + 1, stride):
        for x in range(0, width - window_size[0] + 1, stride):
            window = frame[y:y + window_size[1], x:x + window_size[0]]
            preprocessed_window = preprocess_for_model(window)
            predictions = predict(preprocessed_window)
            class_id = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            if confidence > 0.6:  # Reverted to original confidence threshold
                detections.append([x, y, x + window_size[0], y + window_size[1]])
                scores.append(confidence)

    boxes, final_scores = non_max_suppression(detections, scores, 0.3)

    object_counts = Counter()
    for (x1, y1, x2, y2), score in zip(boxes, final_scores):
        object_img = frame[y1:y2, x1:x2]
        preprocessed_img = preprocess_for_model(object_img)
        predictions = predict(preprocessed_img)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        diameter = calculate_diameter((x1, y1, x2, y2))
        object_counts[predicted_class] += 1

        # Get shelf life information
        shelf_life_info = get_shelf_life_info(predicted_class)
        expiry_date = datetime.now() + timedelta(days=shelf_life_info["estimated_days"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{predicted_class} ({confidence:.2f}) Dia: {diameter}px"
        cv2.putText(frame, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Shelf: {shelf_life_info['shelf']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)
        cv2.putText(frame, f"Refrigerator: {shelf_life_info['refrigerator']}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Use by: {expiry_date.strftime('%Y-%m-%d')}", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)

    quantity_text = ", ".join(f"{k}: {v}" for k, v in object_counts.items())
    cv2.putText(frame, f"Quantities: {quantity_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame


def process_frame(frame_queue, result_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        processed_frame = detect_and_predict(frame)
        result_queue.put(processed_frame)


# Camera initialization
ip_camera_url = "http://192.168.142.196:4747/video"  # Replace <your_ip> and <port> with the actual values
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error: Could not open any camera.")
    exit()

frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

# Start the processing thread
processing_thread = threading.Thread(target=process_frame, args=(frame_queue, result_queue))
processing_thread.start()

last_frame = None
frame_count = 0
process_every_n_frames = 2  # Process every 3rd frame for a balance of performance and accuracy

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    frame_count += 1

    # Process every nth frame
    if frame_count % process_every_n_frames == 0:
        # Put the frame in the queue for processing, replacing any unprocessed frame
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

    # Display the processed frame if available, otherwise display the last processed frame or the original frame
    try:
        display_frame = result_queue.get_nowait()
        last_frame = display_frame
    except queue.Empty:
        if last_frame is not None:
            display_frame = last_frame
        else:
            display_frame = frame

    cv2.imshow('Fruit and Vegetable Quality Detection', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
frame_queue.put(None)  # Signal the processing thread to stop
processing_thread.join()
cap.release()
cv2.destroyAllWindows()