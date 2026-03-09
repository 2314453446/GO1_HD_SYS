#!/usr/bin/env python3

import os
import yaml
import rclpy

from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class CameraLevelTFPublisher(Node):
    def __init__(self):
        super().__init__("camera_level_tf_publisher")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_file = os.path.join(current_dir, "camera_level.yaml")

        if not os.path.exists(yaml_file):
            self.get_logger().error(f"YAML file not found: {yaml_file}")
            raise FileNotFoundError(yaml_file)

        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        parent_frame = data["parent_frame"]
        child_frame = data["child_frame"]
        tx, ty, tz = data["translation"]
        qx, qy, qz, qw = data["rotation"]

        self.br = StaticTransformBroadcaster(self)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        t.transform.translation.x = float(tx)
        t.transform.translation.y = float(ty)
        t.transform.translation.z = float(tz)

        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)

        self.br.sendTransform(t)
        self.get_logger().info(
            f"Published static TF: {parent_frame} -> {child_frame}"
        )


def main():
    rclpy.init()
    node = CameraLevelTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()