from ultralytics import YOLO
import cv2
import numpy as np

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",
    "stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass",
    "cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"
]

class ObjectDetectorAPI:
    def __init__(self, model_path='yolov8m.pt'):
        self.model = YOLO(model_path)
    def predict(self, image, conf_threshold=0.30):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        # Apply confidence filter
        keep = confs > conf_threshold
        boxes = boxes[keep]
        classes = classes[keep]
        confs = confs[keep]

        for i, box in enumerate(boxes.astype(int)):
            class_id = int(classes[i])
            confidence_percent = int(confs[i] * 100)
            label = f"{COCO_CLASSES[class_id]}: {confidence_percent}%"
            #cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            #cv2.putText(image, label, (box[0], box[1]-10),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return image, boxes, classes, confs

