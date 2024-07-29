import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

image_path = 'alik.jpg'
image = cv2.imread(image_path)
height, width, channels = image.shape

blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

confidence_dict = {}
for i in range(len(class_ids)):
    class_name = classes[class_ids[i]]
    confidence = confidences[i]
    if class_name in confidence_dict:
        confidence_dict[class_name].append(confidence)
    else:
        confidence_dict[class_name] = [confidence]

highest_confidence_class = max(confidence_dict, key=lambda k: np.mean(confidence_dict[k]))

highest_confidence_count = len(confidence_dict[highest_confidence_class])

print(f"The object with the highest confidence is '{highest_confidence_class}' which appears {highest_confidence_count} times.")

for i in range(len(boxes)):
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    color = (0, 255, 0) if label == highest_confidence_class else (0, 0, 255)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
