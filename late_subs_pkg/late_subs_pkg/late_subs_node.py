import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class LateSubsNode(Node):
    def __init__(self):
        super().__init__('late_subs_node')

        self.create_subscription(
                Detection2DArray,
                '/late_fusion/yolo/human_detections',
                self._human_detections_callback,
                1)

        self.create_subscription(
                Detection2DArray,
                '/late_fusion/yolo/car_detections',
                self._car_detections_callback,
                1)

        self.get_logger().info("Subscription node running...")

    def _car_detections_callback(self, car_detections_msg: Detection2DArray) -> None:
        """
        Callback to process Detection2DArray messages containing car detections.

        Args:
            car_detections_msg (Detection2DArray): Message containing detected cars.
        """
        detections_info = []

        for detection in car_detections_msg.detections:
            # Cada detección puede tener múltiples hipótesis (normalmente 1 en YOLO)
            for hypothesis in detection.results:
                class_id = hypothesis.hypothesis.class_id
                score = hypothesis.hypothesis.score

                bbox = detection.bbox
                center_x = bbox.center.position.x
                center_y = bbox.center.position.y
                width = bbox.size_x
                height = bbox.size_y

                detections_info.append({
                    'class_id': class_id,
                    'score': score,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height
                })

        # Aquí puedes hacer lo que quieras con la lista detections_info
        # Por ejemplo: imprimirla
        self.get_logger().info(f"Received {len(detections_info)} car detections")
        for i, det in enumerate(detections_info):
            self.get_logger().info(
                f"Detection {i}: Class {det['class_id']} "
                f"with score {det['score']:.2f} at "
                f"(x={det['center_x']:.1f}, y={det['center_y']:.1f}), "
                f"size ({det['width']:.1f}x{det['height']:.1f})"
            )

    def _human_detections_callback(self, human_detections_msg: Detection2DArray) -> None:
        pass


def main(args=None) -> None:
    """
    Initializes and spins the ImageDetectorNode.
    """
    rclpy.init(args=args)
    node = LateSubsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
