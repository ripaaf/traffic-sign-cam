import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoModelForImageClassification
import numpy as np

# Detection model
det_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
det_model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
# Classification model
cls_processor = AutoImageProcessor.from_pretrained("rohansaraswat/TrafficSignsDetection")
cls_model = AutoModelForImageClassification.from_pretrained("rohansaraswat/TrafficSignsDetection")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
det_model.to(device)
cls_model.to(device)

cap = cv2.VideoCapture(0)
cls_labels = cls_model.config.id2label

def is_round(crop):
    # Convert to gray and blur
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    # Hough Circle detection
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=gray.shape[0]//8,
        param1=50, param2=30, minRadius=gray.shape[0]//8, maxRadius=gray.shape[0]//2
    )
    return circles is not None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame, master.")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det_inputs = det_processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        det_outputs = det_model(**det_inputs)

    target_sizes = torch.tensor([img.shape[:2]]).to(device)
    results = det_processor.post_process_object_detection(det_outputs, threshold=0.7, target_sizes=target_sizes)[0]

    for box in results["boxes"]:
        box = box.int().cpu().numpy()
        x1, y1, x2, y2 = box
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = img[y1c:y2c, x1c:x2c]

        # Only analyze if the crop is big enough
        if crop.shape[0] > 10 and crop.shape[1] > 10 and is_round(crop):
            # Only classify if the shape is round
            cls_inputs = cls_processor(images=crop, return_tensors="pt").to(device)
            with torch.no_grad():
                cls_outputs = cls_model(**cls_inputs)
                probs = torch.nn.functional.softmax(cls_outputs.logits, dim=1)
                conf, pred = torch.max(probs, 1)
                label = cls_labels[pred.item()]
                confidence = conf.item()
            cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x1c, y1c - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Optionally, draw a red box for non-round objects (for debugging)
        # else:
        #     cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0, 0, 255), 2)

    cv2.imshow('Round Traffic Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting, master~")
        break

cap.release()
cv2.destroyAllWindows()
