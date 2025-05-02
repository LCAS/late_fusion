#!/usr/bin/env python3

from time import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from mmdet3d.apis import LidarDet3DInferencer
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import QoSProfile, ReliabilityPolicy


class LidarDetectorNode(Node):
    def __init__(self):
        super().__init__('lidar_detector_node')

        # Define QoS profile
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)

        # Declare subscriptor
        self.declare_parameter('pointcloud_topic', '/lidar/points')
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.subscription = self.create_subscription(
                PointCloud2,
                pointcloud_topic,
                self._main_pipeline,
                qos_profile)

        # Declare publisher
        self.declare_parameter('bounding_box_topic', '/detected_bounding_boxes')
        bounding_box_topic = self.get_parameter('bounding_box_topic').value
        self.bbox_publisher = self.create_publisher(MarkerArray, bounding_box_topic, qos_profile)

        # Declare model parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('weights_path', '')
        self.declare_parameter('model_threshold', 0.3)
        self.declare_parameter('device', 'cpu')

        model_path = self.get_parameter('model_path').value
        weights_path = self.get_parameter('weights_path').value
        self.model_threshold = self.get_parameter('model_threshold').value
        device = self.get_parameter('device').value

        self.inferencer = LidarDet3DInferencer(
            model=model_path,
            weights=weights_path,
            device=device
        )

        self.get_logger().info("Lidar detector node running...")

    def _main_pipeline(self, msg):

        points = self.convert_pc2_to_np(msg)

        if points is None:
            return None

        start = time()
        results = self.inferencer({'points': points}, batch_size=1, show=False)
        end = time()

        detections = self.create_marker_array_from_predictions(results)
        self.bbox_publisher.publish(detections)

        self.get_logger().info(f"{len(detections.markers)} detections in {end-start: .4f} s")

    def convert_pc2_to_np(self, lidar_msg):
        return pc2.read_points_numpy(
                lidar_msg,
                field_names=("x", "y", "z", "intensity"))

    def create_marker_array_from_predictions(self, results):
        bbox_data = results['predictions'][0]['bboxes_3d']
        bbox_labels = results['predictions'][0]['labels_3d']
        bbox_scores = results['predictions'][0]['scores_3d']
        marker_array = MarkerArray()

        header = Header()
        header.frame_id = "lidar_frame"
        header.stamp = self.get_clock().now().to_msg()

        label_colors = {
            0: (1.0, 0.0, 0.0),
            1: (0.0, 1.0, 0.0),
            2: (0.0, 0.0, 1.0),
        }

        for i, (bbox, label, score) in enumerate(zip(bbox_data, bbox_labels, bbox_scores)):
            if score < self.model_threshold:
                continue

            r, g, b = label_colors.get(label, (1.0, 1.0, 1.0))

            marker = Marker()
            marker.header = header
            marker.ns = "bounding_boxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = bbox[0]
            marker.pose.position.y = bbox[1]
            marker.pose.position.z = bbox[2]

            marker.scale.x = bbox[3]
            marker.scale.y = bbox[4]
            marker.scale.z = bbox[5]

            marker.color.a = 0.5
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b

            marker_array.markers.append(marker)

        return marker_array


def main(args=None):
    rclpy.init(args=args)
    node = LidarDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
