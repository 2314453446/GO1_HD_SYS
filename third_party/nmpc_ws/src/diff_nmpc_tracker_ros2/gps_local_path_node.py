#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Quaternion
from rclpy.qos import QoSProfile, DurabilityPolicy


class GPSLocalPathNode(Node):
    def __init__(self):
        super().__init__('gps_local_path_node')

        # --- 参数声明 ---
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('path_topic', '/furrow/centerline')
        self.declare_parameter('rviz_goal_topic', '/goal_pose')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('step_size', 0.1)
        self.declare_parameter('turning_radius', 1.0)
        self.declare_parameter('path_mode', 'dubins')
        self.declare_parameter('publish_rate', 10.0)

        self.odom_topic = self.get_parameter('odom_topic').value
        self.path_topic = self.get_parameter('path_topic').value
        self.rviz_goal_topic = self.get_parameter('rviz_goal_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.step_size = float(self.get_parameter('step_size').value)
        self.turning_radius = float(self.get_parameter('turning_radius').value)
        self.path_mode = self.get_parameter('path_mode').value
        self.publish_rate = float(self.get_parameter('publish_rate').value)

        # --- 状态变量 ---
        self.curr_x = None
        self.curr_y = None
        self.curr_yaw = None

        # 用于存储生成的静态路径
        self.static_path_msg = None

        # --- 发布与订阅 ---
        # 使用普通的 QoS，定时器重复发送同一条路径，确保所有节点都能收到
        self.pub_path = self.create_publisher(Path, self.path_topic, 10)

        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self.odom_cb, 10
        )
        self.sub_rviz_goal = self.create_subscription(
            PoseStamped, self.rviz_goal_topic, self.rviz_goal_cb, 10
        )

        # 定时器现在只负责“广播”静态路径，不再参与任何轨迹的“计算”
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_cb)
        self.get_logger().info('Static path node started. Waiting for /odom and RViz goal...')

    # -----------------------------
    # Callbacks
    # -----------------------------
    def odom_cb(self, msg: Odometry):
        # 持续更新机器人当前位置，但仅用于在点击 RViz 目标的那一瞬间获取“起点”
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.curr_yaw = self.yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.frame_id = msg.header.frame_id

    def rviz_goal_cb(self, msg: PoseStamped):
        # 必须先收到过 /odom 知道当前位置，才能作为起点
        if self.curr_x is None or self.curr_yaw is None:
            self.get_logger().warn('Ignored goal: Odom data not received yet.')
            return

        # 1. 提取目标点
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        q = msg.pose.orientation
        goal_yaw = self.yaw_from_quaternion(q.x, q.y, q.z, q.w)

        # 2. 锁定这一瞬间机器人的位置作为起点
        start_x = self.curr_x
        start_y = self.curr_y
        start_yaw = self.curr_yaw

        self.get_logger().info(
            f'Generating STATIC path from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})'
        )

        # 3. 仅在此刻进行一次完整的路径计算
        if self.path_mode == 'dubins':
            px, py, pyaw = self.generate_dubins_path(
                start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, self.turning_radius, self.step_size
            )
        else:
            px, py, pyaw = self.generate_line_path(start_x, start_y, goal_x, goal_y, self.step_size)

        if len(px) < 2:
            self.get_logger().warn('Failed to generate a valid path.')
            return

        # 4. 构建并保存路径消息对象，供定时器反复广播
        path = Path()
        path.header.frame_id = self.frame_id
        
        for i in range(len(px)):
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.pose.position.x = float(px[i])
            ps.pose.position.y = float(py[i])
            ps.pose.position.z = 0.0
            ps.pose.orientation = self.quaternion_from_yaw(pyaw[i])
            path.poses.append(ps)
            
        self.static_path_msg = path

    def timer_cb(self):
        # 如果还没生成过静态路径，就什么都不发
        if self.static_path_msg is None:
            return

        # 仅仅是更新一下时间戳，把保存在内存里的同一条轨迹不断发出去
        # 坐标点数据完全不会发生变化！
        self.static_path_msg.header.stamp = self.get_clock().now().to_msg()
        for pose in self.static_path_msg.poses:
            pose.header.stamp = self.static_path_msg.header.stamp

        self.pub_path.publish(self.static_path_msg)

    # -----------------------------
    # 路径生成核心算法保持不变...
    # -----------------------------
    def generate_line_path(self, sx, sy, gx, gy, ds):
        dist = math.hypot(gx - sx, gy - sy)
        if dist < 1e-6:
            return [sx], [sy], [0.0]

        yaw = math.atan2(gy - sy, gx - sx)
        n = max(2, int(dist / ds) + 1)

        px = np.linspace(sx, gx, n).tolist()
        py = np.linspace(sy, gy, n).tolist()
        pyaw = [yaw] * n
        return px, py, pyaw

    def generate_dubins_path(self, sx, sy, syaw, gx, gy, gyaw, Rmin, ds):
        dx = gx - sx
        dy = gy - sy
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return [sx], [sy], [syaw]

        target_yaw = math.atan2(dy, dx)
        heading_err = self.wrap_angle(target_yaw - syaw)

        arc_len = abs(heading_err) * Rmin
        n_arc = max(2, int(arc_len / ds) + 1)

        px, py, pyaw = [sx], [sy], [syaw]
        x, y, yaw = sx, sy, syaw

        if abs(heading_err) > 1e-3:
            dtheta = heading_err / (n_arc - 1)
            for _ in range(n_arc - 1):
                yaw_mid = yaw + 0.5 * dtheta
                x += Rmin * dtheta * math.cos(yaw_mid)
                y += Rmin * dtheta * math.sin(yaw_mid)
                yaw += dtheta
                px.append(x)
                py.append(y)
                pyaw.append(yaw)

        rem = math.hypot(gx - x, gy - y)
        n_line = max(2, int(rem / ds) + 1)
        for i in range(1, n_line):
            ratio = i / (n_line - 1)
            xx = x + ratio * (gx - x)
            yy = y + ratio * (gy - y)
            yyaw = math.atan2(gy - y, gx - x) if rem > 1e-6 else yaw
            px.append(xx)
            py.append(yy)
            pyaw.append(yyaw)

        px[-1] = gx
        py[-1] = gy
        pyaw[-1] = gyaw if math.isfinite(gyaw) else pyaw[-1]

        return px, py, pyaw

    # -----------------------------
    # Utils
    # -----------------------------
    def yaw_from_quaternion(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def quaternion_from_yaw(self, yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def wrap_angle(self, a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    node = GPSLocalPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

