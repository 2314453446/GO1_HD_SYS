#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diff_nmpc_node.py – NMPC controller for differential-drive model
Controls:  u = [v, ω]  (linear speed, angular speed)

改动：
- 使用 KD-Tree 最近点搜索 + 弧长插值 (np.interp)
- 以弧长 s 均匀采样（ds = v_ref * dt，dt = HORIZON / N_NODE），保证时间一致性
"""

import os, sys, math, timeit
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg     import Odometry, Path

from scipy.spatial import cKDTree

# acados (保持与原工程一致的导入路径)
from acados_template import AcadosOcpSolver, AcadosSimSolver
from diff_nmpc_tracker_ros2.diff_robot_opt   import MobileRobotOptimizer
from diff_nmpc_tracker_ros2.diff_robot_model import MobileRobotModel

# ------------------------- 参数 -------------------------
HORIZON       = 2.0      # [s]
N_NODE        = 100      # 预测离散节点数
CTRL_RATE     = 50.0    # [Hz]
DEFAULT_V_REF = 0.20      # [m/s] 参考速度（可视需要改成ROS参数） 

class NMPCController(Node):
    def __init__(self):
        super().__init__('nmpc_controller')

        # 可选ROS参数（保留与原版相近的风格）
        self.declare_parameter('v_ref', DEFAULT_V_REF)

        # --------------- ACADOS 初始化 -----------------
        model = MobileRobotModel()
        self.opt = MobileRobotOptimizer(
            model.model,
            model.constraint,
            t_horizon=HORIZON,
            n_nodes=N_NODE
        )

        self.x_now = None     # 当前状态 (x, y, theta)
        self.tlog  = []       # 求解时间日志

        # --- 参考轨迹的空间索引容器 ---
        self.ref_ready = False
        self.path_xy   = None      # (N,2) 全局路径离散点
        self.yaw_ref   = None      # (N,)   对应航向角序列
        self.s_acc     = None      # (N,)   弧长累计
        self.kdtree    = None      # KD-Tree on path_xy
        # 只许前进的最近弧长缓存
        self.s0_prev = 0.0

        # ----------------- ROS I/O -----------------
        self.pub_cmd  = self.create_publisher(Twist, '/cmd_vel', 1)
        self.sub_odom = self.create_subscription(Odometry, '/odom',
                                                 self.odom_cb, 1)

        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,  # 改为 VOLATILE 以匹配常见发布端
        )
        self.sub_ref  = self.create_subscription(Path, '/furrow/centerline_odom',
                                                 self.ref_cb, qos)

        # 可视化发布器
        self.exec_path_pub = self.create_publisher(Path, '/exec_path', 10)
        self.pred_path_pub = self.create_publisher(Path, '/pred_path', 10)

        self.exec_path_msg = Path()
        self.exec_path_msg.header.frame_id = 'map'

        # 控制主循环
        self.timer = self.create_timer(1.0 / CTRL_RATE, self.timer_cb)

    # ---------------------- 回调 ----------------------
    def odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        # 从四元数取 yaw
        th = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.x_now = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            th
        ])

    def ref_cb(self, msg: Path):
        """接收 Path 并构建 KD-Tree + 弧长索引"""
        poses = msg.poses
        if not poses:
            self.get_logger().warn("Received empty reference path")
            self.ref_ready = False
            return

        N = len(poses)
        xy  = np.zeros((N, 2), dtype=float)
        yaw_from_msg = np.zeros((N, ),  dtype=float)
        for i, ps in enumerate(poses):
            xy[i, 0] = ps.pose.position.x
            xy[i, 1] = ps.pose.position.y
            q = ps.pose.orientation
            yaw_from_msg[i] = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

        # 计算弧长累计 s_acc
        ds = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        s_acc = np.concatenate(([0.0], np.cumsum(ds)))

        # 保存路径与弧长
        self.path_xy = xy
        self.s_acc   = s_acc

        # 计算几何切线 yaw，使用弧长作为参数，避免噪声
        if N >= 3:
            dx = np.gradient(self.path_xy[:, 0], self.s_acc, edge_order=2)
            dy = np.gradient(self.path_xy[:, 1], self.s_acc, edge_order=2)
            yaw_geo = np.unwrap(np.arctan2(dy, dx))
        elif N == 2:
            dx = self.path_xy[1, 0] - self.path_xy[0, 0]
            dy = self.path_xy[1, 1] - self.path_xy[0, 1]
            ang = math.atan2(dy, dx)
            yaw_geo = np.array([ang, ang])
            self.get_logger().warn("Reference path has only 2 points; using simple segment yaw.")
        else:  # N == 1
            yaw_geo = np.array([0.0])
            self.get_logger().warn("Reference path has only 1 point; yaw set to 0.")
        # 评估消息自带 yaw 的质量，异常则回退到几何 yaw
        yaw_msg_unwrap = np.unwrap(yaw_from_msg)
        if (np.std(np.diff(yaw_from_msg)) < 1e-3) or (not np.isfinite(yaw_from_msg).all()):
            yaw = yaw_geo
        else:
            yaw = yaw_msg_unwrap
        self.yaw_ref = yaw

        # 预缓存 cos/sin 以便后续按弧长插值
        self.c_ref = np.cos(self.yaw_ref)
        self.s_ref = np.sin(self.yaw_ref)

        # 构建 KD-Tree
        self.kdtree  = cKDTree(xy)
        self.ref_ready = True

 # ★★★【新增】重置历史弧长记录，防止第二次运行时跑飞 ★★★
        self.s0_prev = 0.0 
        # ==========================================
        self.get_logger().info(
            f"Reference path loaded: {N} pts, length={s_acc[-1]:.2f} m"
        )

    # ---------------------- 主循环 ----------------------
    def timer_cb(self):
        if not self.ref_ready or self.x_now is None:
            return

        # --- 1) 最近点（KD-Tree） -> 得到弧长 s0 ---
        dist, k = self.kdtree.query(self.x_now[:2])
        s0 = self.s_acc[k]
        # 只许前进：防止弧长回跳
        s0 = max(s0, self.s0_prev)
        self.s0_prev = s0

        # --- 2) 按时间均匀 -> 弧长均匀采样 ---
        # dt 为单步时长，ds = v_ref * dt
        v_ref = float(self.get_parameter('v_ref').value)
        dt_mpc = HORIZON / N_NODE
        ds = max(1e-6, v_ref * dt_mpc)   # 防止零速导致 ds=0

        # 生成 (N+1) 个弧长采样点
        s_vec = s0 + ds * np.arange(self.opt.N + 1, dtype=float)
        # 越界裁剪，防止插值越界
        s_vec = np.clip(s_vec, 0.0, self.s_acc[-1])

        # --- 3) 批量插值 x/y/方向（对 cos/sin 插值，避免 ±π 翻转）---
        x_vec = np.interp(s_vec, self.s_acc, self.path_xy[:, 0])
        y_vec = np.interp(s_vec, self.s_acc, self.path_xy[:, 1])

        cos_th = np.interp(s_vec, self.s_acc, self.c_ref)
        sin_th = np.interp(s_vec, self.s_acc, self.s_ref)
        norm = np.hypot(cos_th, sin_th)
        cos_th = np.divide(cos_th, np.maximum(norm, 1e-9))
        sin_th = np.divide(sin_th, np.maximum(norm, 1e-9))

        # --- 4) 终点减速软着陆（线性收敛） ---
        remain = self.s_acc[-1] - s0
        L_stop = 0.5  # TODO: 可改为 ROS 参数
        scale = float(np.clip(remain / L_stop, 0.0, 1.0))
        v_ref_eff = v_ref * scale

        # --- 5) 设置滚动参考 yref ---
        # 差速模型下，yref 结构为 [x, y, cos(th), sin(th), v_ref, 0.0]
        # 末端（N阶）只给状态参考：[x, y, cos(th), sin(th)]
        for j in range(self.opt.N):
            yref = np.array([x_vec[j], y_vec[j], cos_th[j], sin_th[j],
                             v_ref_eff, 0.0], dtype=float)
            self.opt.solver.set(j, "yref", yref)

        self.opt.solver.set(self.opt.N, "yref",
                            np.array([x_vec[-1], y_vec[-1],
                                      cos_th[-1], sin_th[-1]], dtype=float))

        # --- 6) 当前状态硬约束 x0 ---
        self.opt.solver.set(0, "lbx", self.x_now)
        self.opt.solver.set(0, "ubx", self.x_now)

        # --- 7) 求解 NMPC ---
        tic = timeit.default_timer()
        status = self.opt.solver.solve()
        solve_t = timeit.default_timer() - tic
        self.tlog.append(solve_t)

        if status != 0:
            self.get_logger().warn(f"ACADOS status {status}")
            return

        # 取第一步控制
        u0 = self.opt.solver.get(0, "u")  # [v, ω]

        # --- 8) 发布 /cmd_vel ---
        cmd = Twist()
        cmd.linear.x  = float(u0[0])
        cmd.angular.z = float(u0[1])
        self.pub_cmd.publish(cmd)

        # --- 9) 可视化：预测轨迹与执行轨迹 ---
        self.publish_prediction_path()
        self.log_executed_path()

    # ------------------ 可视化辅助 ------------------
    def publish_prediction_path(self):
        """发布 NMPC 预测状态轨迹"""
        pred = Path()
        t_now = self.get_clock().now().to_msg()
        pred.header.stamp = t_now
        pred.header.frame_id = 'map'

        for i in range(self.opt.N + 1):
            x_pred = self.opt.solver.get(i, "x")  # [x, y, theta]

            pose = PoseStamped()
            pose.header.stamp = t_now
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(x_pred[0])
            pose.pose.position.y = float(x_pred[1])

            theta = float(x_pred[2])
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)

            pred.poses.append(pose)

        self.pred_path_pub.publish(pred)

    def log_executed_path(self):
        """记录并发布执行轨迹"""
        if self.x_now is None:
            return

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(self.x_now[0])
        pose.pose.position.y = float(self.x_now[1])
        pose.pose.orientation.z = math.sin(self.x_now[2] / 2.0)
        pose.pose.orientation.w = math.cos(self.x_now[2] / 2.0)

        self.exec_path_msg.poses.append(pose)
        self.exec_path_pub.publish(self.exec_path_msg)

# ------------------------- 主入口 -------------------------
def main():
    rclpy.init()
    node = NMPCController()
    rclpy.spin(node)

    # 结束打印求解统计
    if node.tlog:
        t = np.array(node.tlog) * 1e3  # ms
        print(f"\nNMPC Solve time: mean={t.mean():.2f} ms | "
              f"max={t.max():.2f} ms | min={t.min():.2f} ms")

if __name__ == '__main__':
    main()
