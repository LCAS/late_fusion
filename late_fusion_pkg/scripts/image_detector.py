from time import time
from typing import Tuple

import torch
from ultralytics import YOLO
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D


class ImageDetector:

    def __init__(self, YOLO_model: str, YOLO_threshold: float, device: str) -> None:

        self.model = YOLO(YOLO_model).to(torch.device(device))
        self.model.fuse()
        self.threshold = YOLO_threshold
        self.classes = [0, 2]

        self.bridge = CvBridge()

    def run(self, image_msg: Image):
        header = image_msg.header

        cv2_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        start = time()
        output_image, predictions = self.detect(cv2_image)
        elapsed = time() - start

        output_image_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
        car_pred_msg, human_pred_msg = self.npbbox2detections(predictions, header)


        return elapsed, len(predictions), car_pred_msg, human_pred_msg, output_image_msg

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = self.model(
            img,
            conf=self.threshold,
            classes=self.classes,
            verbose=False
        )

        img_with_boxes = predictions[0].plot()
        pred_bboxes = (predictions[0].boxes.data).detach().cpu().numpy()

        return img_with_boxes, pred_bboxes

    def npbbox2detections(self, detections: np.ndarray, header) -> Tuple[Detection2DArray, Detection2DArray]:
        cars_msg = Detection2DArray()
        cars_msg.header = header

        humans_msg = Detection2DArray()
        humans_msg.header = header

        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            detection = Detection2D()

            # Define bounding box
            bbox = BoundingBox2D()
            bbox.center.position.x = float((x1 + x2) / 2.0)
            bbox.center.position.y = float((y1 + y2) / 2.0)
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)
            detection.bbox = bbox

            # Define object hypothesis
            hypo = ObjectHypothesisWithPose()
            hypo.hypothesis.class_id = str(class_id)
            hypo.hypothesis.score = float(conf)
            detection.results.append(hypo)

            # Classify detection
            if hypo.hypothesis.class_id == '0.0':
                humans_msg.detections.append(detection)
            elif hypo.hypothesis.class_id == '2.0':
                cars_msg.detections.append(detection)

        return cars_msg, humans_msg
