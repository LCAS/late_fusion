# MIT License
# This file is part of a project licensed under the MIT License.
# See the LICENSE file in the repository for details.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from message_filters import Subscriber, TimeSynchronizer

from scripts.image_detector import ImageDetector


class LateFusionNode(Node):

    def __init__(self) -> None:
        super().__init__('late_fusion_node')

        # Subscription to input images
        self.declare_parameter(
                'image_input_topic',
                '/camera/image_raw')

        image_input_topic = self.get_parameter('image_input_topic').value

        ts = TimeSynchronizer(
                [
                    Subscriber(self, Image, image_input_topic)
                    ],
                queue_size=10
                )
        ts.registerCallback(self._main_pipeline)

        # Publishers for processed images and detections
        self.declare_parameter(
                'YOLO_output_image_topic',
                '/yolo/image')
        self.declare_parameter(
                'YOLO_human_detections_topic',
                '/yolo/human_detections_2d')
        self.declare_parameter(
                'YOLO_car_detections_topic',
                '/yolo/car_detections_2d')

        YOLO_output_image_topic = self.get_parameter(
                'YOLO_output_image_topic').value
        YOLO_human_detections_topic = self.get_parameter(
                'YOLO_human_detections_topic').value
        YOLO_car_detections_topic = self.get_parameter(
                'YOLO_car_detections_topic').value

        self.yolo_image_publisher = self.create_publisher(
                Image,
                YOLO_output_image_topic,
                1)
        self.yolo_humans_publisher = self.create_publisher(
                Detection2DArray,
                YOLO_human_detections_topic,
                1)
        self.yolo_cars_publisher = self.create_publisher(
                Detection2DArray,
                YOLO_car_detections_topic,
                1)

        # YOLO model parameters
        self.declare_parameter('YOLO_model', 'yolov5su.pt')
        self.declare_parameter('YOLO_threshold', 0.5)
        self.declare_parameter('device', 'cpu')

        self.image_detector = ImageDetector(
                self.get_parameter('YOLO_model').value,
                self.get_parameter('YOLO_threshold').value,
                self.get_parameter('device').value,
                )

        self.get_logger().info("Late fusion node is up and running.")

    def _main_pipeline(self, image_msg: Image) -> None:
        elapsed, yolo_pred, car_pred_msg, human_pred_msg, yolo_image_msg = self.image_detector.run(image_msg)
        self.get_logger().info(f"Yolo detected {yolo_pred} objects in {elapsed :.4f}")


        self.yolo_image_publisher.publish(yolo_image_msg)
        self.yolo_humans_publisher.publish(human_pred_msg)
        self.yolo_cars_publisher.publish(car_pred_msg)



def main(args=None) -> None:
    """
    Initializes and spins the ImageDetectorNode.
    """
    rclpy.init(args=args)
    node = LateFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
