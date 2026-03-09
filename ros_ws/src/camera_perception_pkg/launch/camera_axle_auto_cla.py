#!/usr/bin/env python3

import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped

from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation


class CameraGravityCalibration(Node):

    def __init__(self):

        super().__init__("camera_gravity_calibration")

        # IMU 数据缓存
        self.samples = []

        # 采样数量
        self.max_samples = 200

        # 标定完成标志
        self.calibrated = False

        # 发布静态 TF
        self.br = StaticTransformBroadcaster(self)

        # 订阅 IMU
        self.create_subscription(
            Imu,
            "/camera/gyro_accel/sample",
            self.cb_imu,
            100
        )

        self.get_logger().info("Collecting IMU data for gravity calibration...")

    def cb_imu(self, msg):

        if self.calibrated:
            return

        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        g = np.array([ax, ay, az], dtype=np.float32)

        if np.linalg.norm(g) < 1e-6:
            return

        self.samples.append(g)

        if len(self.samples) >= self.max_samples:
            self.compute_calibration()

    def compute_calibration(self):

        self.get_logger().info("Computing camera leveling...")

        # 计算平均重力
        g = np.mean(self.samples, axis=0)

        g = g / np.linalg.norm(g)

        gx, gy, gz = g

        self.get_logger().info(f"gravity vector = {g}")

        # 只绕 Y 轴旋转
        pitch = np.arctan2(gz, -gy)

        self.get_logger().info(
            f"pitch correction = {np.degrees(pitch):.3f} deg"
        )

        # 构建旋转矩阵
        R = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0,             1, 0],
            [-np.sin(pitch),0, np.cos(pitch)]
        ])

        quat = Rotation.from_matrix(R).as_quat()

        # 构建 TF
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()

        # 挂在 camera_link 下
        t.header.frame_id = "camera_link"

        t.child_frame_id = "camera_level_frame"

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        # 发布静态 TF
        self.br.sendTransform(t)

        self.calibrated = True

        self.get_logger().info("Calibration finished.")
        self.get_logger().info("Static TF published:")
        self.get_logger().info("camera_link → camera_level_frame")


def main():

    rclpy.init()

    node = CameraGravityCalibration()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()