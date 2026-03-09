#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge

from message_filters import Subscriber, ApproximateTimeSynchronizer


@dataclass
class CamK:
    fx: float
    fy: float
    cx: float
    cy: float


def camk_from_info(msg: CameraInfo) -> CamK:
    return CamK(float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5]))


class FurrowPathNode(Node):
    def __init__(self):
        super().__init__("furrow_path_node")
        self.bridge = CvBridge()

        # ---------------- Topics ----------------
        self.depth_topic = "/camera/depth/image_raw"
        self.depth_info_topic = "/camera/depth/camera_info"

        self.color_topic = "/camera/color/image_raw"  # 用于 overlay
        self.color_info_topic = "/camera/color/camera_info"

        # ---------------- Params ----------------
        self.depth_min = 0.2
        self.depth_max = 5.0

        # residual->heat
        self.bg_sigma = 30.0
        self.r_max = 0.15

        # denoise
        self.median_ksize = 5  # odd

        # DP smooth
        self.lambda_smooth = 5.0

        # ROI (image coords)
        self.roi_v0_ratio = 0.45
        self.roi_v1_ratio = 0.95
        self.roi_u0_ratio = 0.20
        self.roi_u1_ratio = 0.80

        # Output sampling
        self.sample_step = 8

        # Heatmap visualization
        self.colormap = cv2.COLORMAP_TURBO  # TURBO / JET 都行
        self.overlay_alpha = 0.45           # heat overlay transparency

        # ---------------- State ----------------
        self.depth_k: Optional[CamK] = None
        self.last_color_bgr: Optional[np.ndarray] = None
        self.last_color_header = None

        # ---------------- Subscribers ----------------
        # Depth sync (depth image + depth camera info)
        self.sub_depth = Subscriber(self, Image, self.depth_topic)
        self.sub_dinfo = Subscriber(self, CameraInfo, self.depth_info_topic)
        self.sync_depth = ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_dinfo],
            queue_size=10,
            slop=0.08
        )
        self.sync_depth.registerCallback(self.cb_depth)

        # Color (不强求严格同步：缓存最新一帧用于 overlay)
        self.create_subscription(Image, self.color_topic, self.cb_color, 10)

        # ---------------- Publishers ----------------
        self.pub_heat = self.create_publisher(Image, "/furrow/heatmap", 10)                # mono8
        self.pub_heat_color = self.create_publisher(Image, "/furrow/heatmap_color", 10)    # bgr8
        self.pub_overlay = self.create_publisher(Image, "/furrow/overlay", 10)             # bgr8 overlay

        self.pub_path = self.create_publisher(Path, "/furrow/centerline", 10)
        self.pub_mk = self.create_publisher(Marker, "/furrow/marker", 10)
        self.pub_conf = self.create_publisher(Float32, "/furrow/confidence", 10)

        self.get_logger().info("FurrowPathNode started.")
        self.get_logger().info("Visualize in RViz2 or rqt_image_view:")
        self.get_logger().info("  /furrow/heatmap (mono8), /furrow/heatmap_color (bgr8), /furrow/overlay (bgr8)")

    # ---------------- Color cache ----------------
    def cb_color(self, msg: Image):
        try:
            # Orbbec/ROS color often "rgb8" or "bgr8"
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            try:
                cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                # try best-effort convert
                if cv.ndim == 3 and cv.shape[2] == 3:
                    cv = cv.copy()
                else:
                    return
            except Exception:
                return

        self.last_color_bgr = cv
        self.last_color_header = msg.header

    # ---------------- Main depth callback ----------------
    def cb_depth(self, depth_msg: Image, info_msg: CameraInfo):
        self.depth_k = camk_from_info(info_msg)

        D = self.depth_to_meters(depth_msg)
        if D is None:
            return

        H, (u_list, v_list), conf = self.compute_centerline_heat_dp(D)

        # ---- heatmap mono8 ----
        heat_u8 = (np.clip(H, 0.0, 1.0) * 255.0).astype(np.uint8)
        heat_msg = self.bridge.cv2_to_imgmsg(heat_u8, encoding="mono8")
        heat_msg.header = depth_msg.header
        self.pub_heat.publish(heat_msg)

        # ---- heatmap pseudo-color bgr8 ----
        heat_color = cv2.applyColorMap(heat_u8, self.colormap)
        heat_color_msg = self.bridge.cv2_to_imgmsg(heat_color, encoding="bgr8")
        heat_color_msg.header = depth_msg.header
        self.pub_heat_color.publish(heat_color_msg)

        # ---- overlay on RGB (if available) ----
        if self.last_color_bgr is not None:
            overlay = self.make_overlay(self.last_color_bgr, heat_color)
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            # 用 color header 更合理（显示时和RGB一致）；也可以用 depth header
            overlay_msg.header = self.last_color_header if self.last_color_header is not None else depth_msg.header
            self.pub_overlay.publish(overlay_msg)

        # ---- confidence ----
        self.pub_conf.publish(Float32(data=float(conf)))

        # ---- build 3D points in *depth frame_id* (optical frame) ----
        pts3d = self.uv_to_3d(D, u_list, v_list, self.depth_k, step=self.sample_step)

        # ---- publish path + marker (frame = depth_msg.header.frame_id) ----
        self.pub_path.publish(self.make_path(depth_msg.header, pts3d))
        self.pub_mk.publish(self.make_marker(depth_msg.header, pts3d))

    # ---------------- Helpers ----------------
    def depth_to_meters(self, msg: Image) -> Optional[np.ndarray]:
        try:
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return None

        if cv.dtype == np.uint16:
            D = cv.astype(np.float32) / 1000.0  # mm -> m
        else:
            D = cv.astype(np.float32)

        D[(D < self.depth_min) | (D > self.depth_max)] = np.nan
        return D

    def compute_centerline_heat_dp(self, D: np.ndarray) -> Tuple[np.ndarray, Tuple[List[int], List[int]], float]:
        h, w = D.shape
        v0 = int(h * self.roi_v0_ratio)
        v1 = int(h * self.roi_v1_ratio)
        u0 = int(w * self.roi_u0_ratio)
        u1 = int(w * self.roi_u1_ratio)

        # fill NaN
        D_f = D.copy()
        D_f[np.isnan(D_f)] = 0.0

        # medianBlur robust way (uint16)
        try:
            D_u16 = np.clip(D_f * 1000.0, 0, 65535).astype(np.uint16)
            D_u16 = cv2.medianBlur(D_u16, self.median_ksize)
            D_f = D_u16.astype(np.float32) / 1000.0
        except Exception as e:
            self.get_logger().warn(f"medianBlur failed (skip): {e}")

        # background trend
        D_bg = cv2.GaussianBlur(D_f, ksize=(0, 0), sigmaX=self.bg_sigma, sigmaY=self.bg_sigma)

        # residual (relative deeper => larger residual)
        R = np.maximum(D_f - D_bg, 0.0)
        R = np.clip(R, 0.0, self.r_max)

        # heat 0..1
        H = R / self.r_max

        # ROI mask
        mask = np.zeros_like(H, dtype=np.float32)
        mask[v0:v1, u0:u1] = 1.0
        H *= mask

        # DP search per row
        rows = list(range(v0, v1))
        cols = np.arange(u0, u1)
        n_rows = len(rows)
        n_cols = len(cols)

        dp = np.full((n_rows, n_cols), np.inf, dtype=np.float32)
        prev = np.full((n_rows, n_cols), -1, dtype=np.int32)

        dp[0, :] = -H[rows[0], cols]
        lam = float(self.lambda_smooth)

        for ri in range(1, n_rows):
            v = rows[ri]
            heat = H[v, cols]
            for ci in range(n_cols):
                du = np.abs(cols[ci] - cols)
                cost = dp[ri - 1, :] + lam * du
                bp = int(np.argmin(cost))
                dp[ri, ci] = cost[bp] - float(heat[ci])
                prev[ri, ci] = bp

        # backtrack
        end_ci = int(np.argmin(dp[-1, :]))
        u_idx = [end_ci]
        for ri in range(n_rows - 1, 0, -1):
            end_ci = int(prev[ri, end_ci])
            u_idx.append(end_ci)
        u_idx.reverse()

        u_list = [int(cols[i]) for i in u_idx]
        v_list = rows

        # confidence: mean heat along path
        conf = float(np.mean([H[v_list[i], u_list[i]] for i in range(n_rows)])) if n_rows else 0.0
        return H, (u_list, v_list), conf

    def uv_to_3d(self, D: np.ndarray, u_list: List[int], v_list: List[int], K: CamK, step: int = 8) -> List[Tuple[float, float, float]]:
        pts = []
        for i in range(0, len(v_list), step):
            u = u_list[i]
            v = v_list[i]
            z = float(D[v, u])
            if not np.isfinite(z) or z <= 0.0:
                continue
            x = (u - K.cx) * z / K.fx
            y = (v - K.cy) * z / K.fy
            # points are in depth_msg.header.frame_id (optical frame): X right, Y down, Z forward
            pts.append((x, y, z))
        return pts

    def make_path(self, header, pts: List[Tuple[float, float, float]]) -> Path:
        path = Path()
        path.header = header
        for (x, y, z) in pts:
            ps = PoseStamped()
            ps.header = header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(z)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

    def make_marker(self, header, pts: List[Tuple[float, float, float]]) -> Marker:
        mk = Marker()
        mk.header = header
        mk.ns = "furrow"
        mk.id = 0
        mk.type = Marker.LINE_STRIP
        mk.action = Marker.ADD
        mk.scale.x = 0.03
        mk.color.a = 1.0
        mk.color.r = 1.0
        mk.color.g = 0.2
        mk.color.b = 0.2

        for (x, y, z) in pts:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = float(z)
            mk.points.append(p)
        return mk

    def make_overlay(self, color_bgr: np.ndarray, heat_color_bgr: np.ndarray) -> np.ndarray:
        """
        Overlay heatmap on RGB for visualization.
        - Resize heat to match color if needed.
        - alpha blend: overlay = (1-a)*color + a*heat
        """
        hC, wC = color_bgr.shape[:2]
        hH, wH = heat_color_bgr.shape[:2]
        if (hC != hH) or (wC != wH):
            heat_rs = cv2.resize(heat_color_bgr, (wC, hC), interpolation=cv2.INTER_NEAREST)
        else:
            heat_rs = heat_color_bgr

        a = float(self.overlay_alpha)
        overlay = cv2.addWeighted(color_bgr, 1.0 - a, heat_rs, a, 0.0)
        return overlay


def main():
    rclpy.init()
    node = FurrowPathNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()