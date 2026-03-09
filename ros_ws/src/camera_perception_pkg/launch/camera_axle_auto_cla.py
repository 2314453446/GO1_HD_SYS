#!/usr/bin/env python3

import rclpy
import numpy as np
import yaml
import os

from rclpy.node import Node
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation


class CameraGravityCalibration(Node):

    def __init__(self):

        super().__init__("camera_gravity_calibration")

        # IMU缓存
        self.samples = []

        # 采样数量
        self.max_samples = 200

        self.calibrated = False

        # 当前脚本目录
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 保存文件
        self.save_file = os.path.join(script_dir, "camera_level.yaml")

        # 订阅IMU
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

        g = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=np.float32)

        if np.linalg.norm(g) < 1e-6:
            return

        self.samples.append(g)

        if len(self.samples) >= self.max_samples:
            self.compute_calibration()

    def compute_calibration(self):

        self.get_logger().info("Computing camera leveling...")

        g = np.mean(self.samples, axis=0)

        g = g / np.linalg.norm(g)

        gx, gy, gz = g

        self.get_logger().info(f"gravity vector = {g}")

        # 只绕Y轴旋转
        pitch = np.arctan2(gz, -gy)

        self.get_logger().info(
            f"pitch correction = {np.degrees(pitch):.3f} deg"
        )

        # 旋转矩阵
        R = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0,             1, 0],
            [-np.sin(pitch),0, np.cos(pitch)]
        ])

        quat = Rotation.from_matrix(R).as_quat()

        # 写入yaml
        data = {
            "parent_frame": "camera_link",
            "child_frame": "camera_level_frame",
            "translation": [0.0, 0.0, 0.0],
            "rotation": [
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
                float(quat[3])
            ]
        }

        with open(self.save_file, "w") as f:
            yaml.dump(data, f)

        self.get_logger().info("Calibration finished.")
        self.get_logger().info(f"Saved calibration to: {self.save_file}")

        self.calibrated = True


def main():

    rclpy.init()

    node = CameraGravityCalibration()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()