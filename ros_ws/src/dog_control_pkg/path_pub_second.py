#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, List, Tuple


import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)
#沟壑是一条完美的直线时，当尝试用五次多项式去强行拟合一条直线时，防止Numpy 可能会在终端里狂刷警告提示矩阵秩不足

import cv2


import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration


from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge


from message_filters import Subscriber, ApproximateTimeSynchronizer


import tf2_ros
from tf2_ros import TransformException
import math


from nav_msgs.msg import Odometry


@dataclass
class CamK:
    fx: float
    fy: float
    cx: float
    cy: float



def camk_from_info(msg: CameraInfo) -> CamK:
    return CamK(float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5]))



def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)


    qx /= n
    qy /= n
    qz /= n
    qw /= n


    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz


    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)]
    ], dtype=np.float64)



def wrap_angle(a: float) -> float:
    return np.arctan2(np.sin(a), np.cos(a))



class FurrowPathNode(Node):


    def __init__(self):
        super().__init__("furrow_path_node")
        self.bridge = CvBridge()


        # ---------------- Topics ----------------
        self.depth_topic = "/camera/depth/image_raw"          # 深度图输入话题
        self.depth_info_topic = "/camera/depth/camera_info"  # 深度相机内参话题
        self.color_topic = "/camera/color/image_raw"         # 彩色图话题，仅用于可视化叠加


        # ---------------- Target frame ----------------
        self.target_frame = "camera_level_frame"             # 重力对齐坐标系；最终路径/目标点都发布到该坐标系下
        # self.target_frame = "odom"
         
        # ---------------- Perception params ----------------
        self.depth_min = 0.2   # 有效最小深度（m），小于该值的深度视为无效
        self.depth_max = 5.0   # 有效最大深度（m），大于该值的深度视为无效


        self.bg_sigma = 22.0   # 高斯背景趋势平滑尺度；越大越强调大尺度地形趋势 可调（22）
        self.r_max = 0.05      # 沟壑相对低点残差上限（m）；热图归一化时的最大深度差
        self.median_ksize = 7  # 中值滤波核大小；用于平滑高度图，必须为奇数 可调（7）


        # ROI：只在图像中间偏下区域寻找沟壑引导线
        self.roi_v0_ratio = 0.45  # ROI 上边界，占图像高度比例
        self.roi_v1_ratio = 0.95  # ROI 下边界，占图像高度比例
        self.roi_u0_ratio = 0.20  # ROI 左边界，占图像宽度比例
        self.roi_u1_ratio = 0.80  # ROI 右边界，占图像宽度比例


        # visualization
        self.colormap = cv2.COLORMAP_TURBO  # 热图伪彩色映射方式
        self.overlay_alpha = 0.45           # 热图叠加到 RGB 上的透明度


        # deeper direction
        self.deeper_is_lower_z = True
        # True: 在 camera_level_frame 中，z 更小表示沟壑更低
        # False: 在你的坐标定义里如果相反，则改成 False


        # ---------------- Visualization tuning ----------------
        self.vis_use_z_window = False
        # 是否只对指定 z 高度范围内的点进行热图构建/可视化
        # 调试阶段建议 False，避免把有效区域裁掉


        self.vis_z_min = -1.2   # 当启用 z 窗口时，允许的最小 z（m）
        self.vis_z_max = -0.05  # 当启用 z 窗口时，允许的最大 z（m）


        self.vis_percentile_low = 5.0   # 彩色热图自适应拉伸下百分位
        self.vis_percentile_high = 98.0 # 彩色热图自适应拉伸上百分位
        # 用于增强可视化对比度，避免少量极值拉坏整体颜色分布


        self.vis_alpha_threshold = 0.10
        # overlay 叠加阈值；热图值低于该值时，不叠加到彩色图上


        # ---------------- Forward-first guidance params ----------------
        self.num_guidance_layers = 20
        # 前向引导层数：把 ROI 从近到远分成多少层来逐层选点
        # 越大，引导线越密；越小，计算更稳更粗


        self.search_half_width = 300
        # 每一层只在参考列左右多少像素内搜索
        # 决定允许多大横向偏移；越大越容易“找侧边低点”，越小越倾向直行


        self.forward_bias_weight = 0.35
        # 正前方偏置权重；越大越优先沿当前前进方向走，不轻易转向


        self.deviation_weight = 0.03
        # 偏离参考列的惩罚权重；越大越不愿意横向偏移


        self.forward_keep_margin = 0.88
        # 如果正前方热度 >= 局部最佳热度 * 0.88，则直接保持前进
        # 越接近 1，越偏向“只要前面差不多好就继续直走”


        self.max_step_u = 50
        # 相邻两层之间允许的最大横向跳变（像素）
        # 用于抑制路径在图像中突然左右跳动


        # ---------------- Stability / control smoothing ----------------
        self.min_confidence = 0.01
        # 当前帧路径被接受的最低置信度
        # 越大越严格，越小越容易接受当前帧路径

        self.points_per_meter = 20.0
        # 五次多项式拟合的重采样密度（每米生成多少个点）
        # 密度可调：增大可让控制器获得更密集的轨迹点

        self.freeze_frames = 8
        # 当当前帧无效时，最多连续沿用上一帧稳定路径的帧数 可调（10-12）


        self.path_num_points = 25
        # 稳定路径统一重采样后的点数
        # 便于时序滤波与控制


        self.path_ema_beta = 0.82
        # 路径 EMA 平滑系数
        # 新稳定路径 = beta * 旧路径 + (1-beta) * 当前路径
        # 越大越稳，但响应越慢


        self.lookahead_distance = 0.5
        # 前视目标点距离（m）
        # 控制器可优先跟踪该距离处的目标点，而不是整条路径


        self.target_ema_beta = 0.80
        # 前视目标点 EMA 平滑系数
        # 越大越稳，越小越灵敏


        self.heading_segment_distance = 0.4
        # 计算 heading 时使用的前视距离（m）
        # 用起点到该距离处点的方向估计当前引导方向


        self.max_heading_step_deg = 3.0
        # heading 每帧最大允许变化角度（度）
        # 防止视觉抖动导致机器人突然大幅转向


        # ---------------- State ----------------
        self.depth_k: Optional[CamK] = None                  # 当前深度相机内参
        self.last_color_bgr: Optional[np.ndarray] = None    # 最近一帧彩色图缓存
        self.last_color_header = None                       # 最近一帧彩色图 header


        self.prev_stable_path: Optional[np.ndarray] = None  # 上一帧稳定路径（N,3）
        self.prev_target: Optional[np.ndarray] = None       # 上一帧稳定前视目标点（3,)
        self.prev_heading: Optional[float] = None           # 上一帧 heading（rad）
        self.prev_u_ref: Optional[float] = None             # 上一帧参考列位置（图像 u）
        self.lost_count = 0                                 # 连续丢失有效路径的帧数


        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()                   # TF 缓存
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # 用于查询深度帧 -> camera_level_frame 的变换


        # ---------------- Subscribers ----------------
        self.sub_depth = Subscriber(self, Image, self.depth_topic)
        self.sub_dinfo = Subscriber(self, CameraInfo, self.depth_info_topic)
        self.sync_depth = ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_dinfo],
            queue_size=10,   # 同步队列长度
            slop=0.08        # 允许的时间差（秒）
        )
        self.sync_depth.registerCallback(self.cb_depth)


        self.create_subscription(Image, self.color_topic, self.cb_color, 10)


        # ---------------- Publishers ----------------
        self.pub_heat = self.create_publisher(Image, "/furrow/heatmap", 10)             # 灰度热图
        self.pub_heat_color = self.create_publisher(Image, "/furrow/heatmap_color", 10) # 彩色热图
        self.pub_overlay = self.create_publisher(Image, "/furrow/overlay", 10)          # RGB+热图叠加图


        self.pub_path = self.create_publisher(Path, "/furrow/centerline", 10)           # 稳定后的引导路径
        self.pub_mk = self.create_publisher(Marker, "/furrow/marker", 10)               # 引导路径 marker
        self.pub_conf = self.create_publisher(Float32, "/furrow/confidence", 10)        # 当前路径置信度
        self.pub_heading = self.create_publisher(Float32, "/furrow/heading", 10)        # 当前引导 heading（rad）
        self.pub_target_mk = self.create_publisher(Marker, "/furrow/target_marker", 10) # 前视目标点 marker


        self.get_logger().info("FurrowPathNode started.")
        self.get_logger().info(f"Gravity-aligned target frame: {self.target_frame}")


        # ================= 新增部分：Odom 转换与发布 =================
        self.pub_path_odom = self.create_publisher(Path, "/furrow/centerline_odom", 10) # odom坐标系下的路径

        # 用于保存 camera_link 在 odom 下的 3D 位姿
        self.cl_odom_R = np.eye(3, dtype=np.float64)
        self.cl_odom_t = np.zeros(3, dtype=np.float64)
        self.cl_pose_ready = False

        # 订阅 /camera_link 获取里程计信息 (假定该话题类型为 Odometry)
        self.create_subscription(Odometry, '/camera_link', self.camera_link_cb, 10)
        # =========================================================

        # # ----------------odom ----------------
        # self.target_frame = "odom"  # 强烈注意：把这里改成 odom！因为我们要发布 odom 下的路径了
        
        # # 新增变量保存最新的 odom
        # self.latest_odom_x = 0.0
        # self.latest_odom_y = 0.0
        # self.latest_odom_yaw = 0.0


        # # 新增对 /odom 的订阅
        # self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

    # ---------------- Camera Link Pose Callback ----------------
    def camera_link_cb(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        # 实时保存 camera_link 到 odom 的平移和旋转矩阵
        self.cl_odom_t = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        self.cl_odom_R = quat_to_rotmat(ori.x, ori.y, ori.z, ori.w)
        self.cl_pose_ready = True


    # ---------------- Color cache ----------------
    def cb_color(self, msg: Image):
        try:
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            try:
                cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if cv.ndim == 3 and cv.shape[2] == 3:
                    cv = cv.copy()
                else:
                    return
            except Exception:
                return


        self.last_color_bgr = cv
        self.last_color_header = msg.header


    # ---------------- Main callback ----------------
    def cb_depth(self, depth_msg: Image, info_msg: CameraInfo):
        self.depth_k = camk_from_info(info_msg)


        D = self.depth_to_meters(depth_msg)
        if D is None:
            return


        src_frame = depth_msg.header.frame_id
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.target_frame,
                src_frame,
                Time(),
                timeout=Duration(seconds=0.2)
            )
        except TransformException as e:
            self.get_logger().warn(
                f"TF lookup failed: {src_frame} -> {self.target_frame}: {e}"
            )
            return


        R, t = self.transform_to_rt(tf_msg)

        # ================= 新增部分：查询到 camera_link 的 TF =================
        try:
            # 查询从 camera_level_frame(source) 到 camera_link(target) 的变换
            tf_cl = self.tf_buffer.lookup_transform(
                "camera_link",
                self.target_frame,
                Time(),
                timeout=Duration(seconds=0.2)
            )
            R_cl, t_cl = self.transform_to_rt(tf_cl)
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {self.target_frame} -> camera_link: {e}")
            return
        # ====================================================================


        H, Z_level, vis_mask, (u_list, v_list), conf = self.compute_forward_guidance_heat_and_line(
            D, self.depth_k, R, t
        )


        # ---- heatmap ----
        heat_u8 = (np.clip(H, 0.0, 1.0) * 255.0).astype(np.uint8)
        heat_msg = self.bridge.cv2_to_imgmsg(heat_u8, encoding="mono8")
        heat_msg.header = depth_msg.header
        self.pub_heat.publish(heat_msg)


        heat_color = self.make_adaptive_heat_color(H, vis_mask)
        heat_color_msg = self.bridge.cv2_to_imgmsg(heat_color, encoding="bgr8")
        heat_color_msg.header = depth_msg.header
        self.pub_heat_color.publish(heat_color_msg)


        if self.last_color_bgr is not None:
            overlay = self.make_overlay(self.last_color_bgr, heat_color, H, vis_mask)
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            overlay_msg.header = self.last_color_header if self.last_color_header is not None else depth_msg.header
            self.pub_overlay.publish(overlay_msg)


        self.pub_conf.publish(Float32(data=float(conf)))


        # ---- raw guidance line in camera_level_frame ----
        raw_pts = self.uv_to_3d_level(
            D, u_list, v_list, self.depth_k, R, t, step=1
        )

        stable_pts = self.update_stable_path(raw_pts, conf)

        # ================= 新增：五次多项式拟合与按密度重采样 =================
        stable_pts = self.fit_and_resample_poly5(stable_pts)
        # =================================================================


        # 1. 转换到 camera_link 坐标系 (通过 TF)
        stable_pts_cl = []
        for x, y, z in stable_pts:
            p = np.array([x, y, z], dtype=np.float64)
            p_cl = R_cl @ p + t_cl
            stable_pts_cl.append((float(p_cl[0]), float(p_cl[1]), float(p_cl[2])))

        # 2. 转换到 odom 坐标系 (通过订阅的 Odometry)
        stable_pts_odom = []
        if self.cl_pose_ready:
            for x, y, z in stable_pts_cl:
                p = np.array([x, y, z], dtype=np.float64)
                p_odom = self.cl_odom_R @ p + self.cl_odom_t
                stable_pts_odom.append((float(p_odom[0]), float(p_odom[1]), float(p_odom[2])))

        out_header = depth_msg.header
        
        # --- 发布 1：camera_link 坐标系下的路径 ---
        out_header.frame_id = "camera_link"
        self.pub_path.publish(self.make_path(out_header, stable_pts_cl))
        self.pub_mk.publish(self.make_marker(out_header, stable_pts_cl))

        # --- 发布 2：odom 坐标系下的路径 ---
        if self.cl_pose_ready:
            out_header.frame_id = "odom"
            self.pub_path_odom.publish(self.make_path(out_header, stable_pts_odom))
            
        # 恢复 frame_id 为 self.target_frame，以免影响下方 compute_heading / target_pt 的发布逻辑
        out_header.frame_id = self.target_frame



        heading = self.compute_heading(stable_pts)
        if heading is not None:
            self.pub_heading.publish(Float32(data=float(heading)))


        target_pt = self.compute_lookahead_target(stable_pts, self.lookahead_distance)
        if target_pt is not None:
            target_pt = self.update_stable_target(target_pt)
            self.pub_target_mk.publish(self.make_target_marker(out_header, target_pt))


    # ---------------- Stability logic ----------------
    def update_stable_path(self, raw_pts: List[Tuple[float, float, float]], conf: float) -> List[Tuple[float, float, float]]:
        valid_measurement = (conf >= self.min_confidence) and (len(raw_pts) >= 4)


        if valid_measurement:
            raw_arr = np.asarray(raw_pts, dtype=np.float64)
            raw_resampled = self.resample_polyline(raw_arr, self.path_num_points)


            if raw_resampled is None:
                valid_measurement = False
            else:
                if self.prev_stable_path is None:
                    stable = raw_resampled
                else:
                    stable = (
                        self.path_ema_beta * self.prev_stable_path
                        + (1.0 - self.path_ema_beta) * raw_resampled
                    )


                self.prev_stable_path = stable
                self.lost_count = 0
        if not valid_measurement:
            if self.prev_stable_path is not None and self.lost_count < self.freeze_frames:
                stable = self.prev_stable_path
                self.lost_count += 1
            else:
                stable = None
                self.prev_target = None
                self.prev_heading = None


        if stable is None:
            return []


        return [(float(p[0]), float(p[1]), float(p[2])) for p in stable]

    def fit_and_resample_poly5(self, pts: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """利用累积弧长进行五次多项式参数化拟合，并按设定密度重采样"""
        if not pts or len(pts) < 6: # 五次多项式至少需要6个点
            return pts

        arr = np.array(pts, dtype=np.float64)
        
        # 1. 计算轨迹的累积弧长 s 作为自变量
        seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total_len = s[-1]

        if total_len < 1e-4:
            return pts

        # 2. 根据用户设置的密度计算需要生成的总点数
        num_points = max(2, int(total_len * self.points_per_meter))

        # 3. 沿弧长 s 进行 5 次多项式拟合 (分别拟合 X, Y, Z 三个维度)
        # 使用多项式拟合能抹平所有视觉毛刺，让路径变得极度顺滑
        poly_x = np.poly1d(np.polyfit(s, arr[:, 0], 5))
        poly_y = np.poly1d(np.polyfit(s, arr[:, 1], 5))
        poly_z = np.poly1d(np.polyfit(s, arr[:, 2], 5))

        # 4. 生成新的均匀弧长并在平滑曲线上采样
        s_new = np.linspace(0, total_len, num_points)
        
        out_pts = []
        for s_val in s_new:
            out_pts.append((float(poly_x(s_val)), float(poly_y(s_val)), float(poly_z(s_val))))

        return out_pts

    def resample_polyline(self, pts: np.ndarray, num_points: int) -> Optional[np.ndarray]:
        if pts is None or len(pts) < 2:
            return None


        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]


        if total < 1e-4:
            return None


        q = np.linspace(0.0, total, num_points)
        out = np.zeros((num_points, 3), dtype=np.float64)


        for d in range(3):
            out[:, d] = np.interp(q, s, pts[:, d])


        return out


    def compute_lookahead_target(self, pts: List[Tuple[float, float, float]], lookahead_distance: float) -> Optional[np.ndarray]:
        if pts is None or len(pts) < 2:
            return None


        arr = np.asarray(pts, dtype=np.float64)
        seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        if total < 1e-4:
            return arr[0].copy()


        target_s = min(lookahead_distance, total)


        out = np.zeros(3, dtype=np.float64)
        for d in range(3):
            out[d] = np.interp(target_s, s, arr[:, d])
        return out


    def update_stable_target(self, target_raw: np.ndarray) -> np.ndarray:
        if self.prev_target is None:
            self.prev_target = target_raw.copy()
        else:
            self.prev_target = (
                self.target_ema_beta * self.prev_target
                + (1.0 - self.target_ema_beta) * target_raw
            )
        return self.prev_target.copy()


    def compute_heading(self, pts: List[Tuple[float, float, float]]) -> Optional[float]:
        if pts is None or len(pts) < 2:
            return None


        arr = np.asarray(pts, dtype=np.float64)
        target_pt = self.compute_lookahead_target(pts, self.heading_segment_distance)
        if target_pt is None:
            return None


        p0 = arr[0]
        v = target_pt[:2] - p0[:2]
        n = np.linalg.norm(v)
        if n < 1e-6:
            return self.prev_heading


        raw_heading = float(np.arctan2(v[1], v[0]))


        if self.prev_heading is None:
            heading = raw_heading
        else:
            max_step = np.deg2rad(self.max_heading_step_deg)
            diff = wrap_angle(raw_heading - self.prev_heading)
            diff = np.clip(diff, -max_step, max_step)
            heading = wrap_angle(self.prev_heading + diff)


        self.prev_heading = heading
        return heading


    # ---------------- TF helpers ----------------
    def transform_to_rt(self, tf_msg) -> Tuple[np.ndarray, np.ndarray]:
        qx = tf_msg.transform.rotation.x
        qy = tf_msg.transform.rotation.y
        qz = tf_msg.transform.rotation.z
        qw = tf_msg.transform.rotation.w


        tx = tf_msg.transform.translation.x
        ty = tf_msg.transform.translation.y
        tz = tf_msg.transform.translation.z


        R = quat_to_rotmat(qx, qy, qz, qw)
        t = np.array([tx, ty, tz], dtype=np.float64)
        return R, t


    # ---------------- Depth helpers ----------------
    def depth_to_meters(self, msg: Image) -> Optional[np.ndarray]:
        try:
            cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return None


        if cv.dtype == np.uint16:
            D = cv.astype(np.float32) / 1000.0
        else:
            D = cv.astype(np.float32)


        D[(D < self.depth_min) | (D > self.depth_max)] = np.nan
        return D


    # ---------------- Forward-first guidance core ----------------
    def compute_forward_guidance_heat_and_line(
        self,
        D: np.ndarray,
        K: CamK,
        R: np.ndarray,
        t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[List[int], List[int]], float]:
        """
        先构建基于 camera_level_frame z 的沟壑热图 H，
        再采用“正前方优先 + 局部低点引导”的逐层推进策略生成引导线。
        """
        h, w = D.shape
        v0 = int(h * self.roi_v0_ratio)
        v1 = int(h * self.roi_v1_ratio)
        u0 = int(w * self.roi_u0_ratio)
        u1 = int(w * self.roi_u1_ratio)


        Z_level = np.full((h, w), np.nan, dtype=np.float32)
        vis_mask = np.zeros((h, w), dtype=np.uint8)


        D_roi = D[v0:v1, u0:u1]
        valid = np.isfinite(D_roi) & (D_roi > 0.0)


        if not np.any(valid):
            H = np.zeros_like(D, dtype=np.float32)
            return H, Z_level, vis_mask, ([], []), 0.0


        vv, uu = np.indices(D_roi.shape)
        uu = uu + u0
        vv = vv + v0


        z = D_roi[valid].astype(np.float64)
        u = uu[valid].astype(np.float64)
        v = vv[valid].astype(np.float64)


        x = (u - K.cx) * z / K.fx
        y = (v - K.cy) * z / K.fy
        pts = np.stack([x, y, z], axis=1)


        pts_level = (R @ pts.T).T + t.reshape(1, 3)
        z_level = pts_level[:, 2].astype(np.float32)


        Z_roi = Z_level[v0:v1, u0:u1]
        Z_roi[valid] = z_level
        Z_level[v0:v1, u0:u1] = Z_roi


        # ---- z window mask ----
        if self.vis_use_z_window:
            z_ok = (z_level >= self.vis_z_min) & (z_level <= self.vis_z_max)
        else:
            z_ok = np.ones_like(z_level, dtype=bool)


        vis_roi = vis_mask[v0:v1, u0:u1]
        tmp = np.zeros_like(D_roi, dtype=np.uint8)
        tmp[valid] = z_ok.astype(np.uint8)
        vis_roi[:, :] = tmp
        vis_mask[v0:v1, u0:u1] = vis_roi


        # ---- fill / smooth ----
        Z_f = Z_level.copy()
        if np.any(np.isfinite(Z_roi)):
            fill_value = float(np.nanmedian(Z_roi))
        else:
            fill_value = 0.0
        Z_f[np.isnan(Z_f)] = fill_value


        try:
            Z_mm = Z_f * 1000.0
            Z_mm = np.clip(Z_mm, -32768, 32767).astype(np.int16)
            Z_mm_u16 = (Z_mm.astype(np.int32) + 32768).astype(np.uint16)
            Z_mm_u16 = cv2.medianBlur(Z_mm_u16, self.median_ksize)
            Z_f = (Z_mm_u16.astype(np.int32) - 32768).astype(np.float32) / 1000.0
        except Exception as e:
            self.get_logger().warn(f"medianBlur failed (skip): {e}")


        Z_bg = cv2.GaussianBlur(Z_f, ksize=(0, 0), sigmaX=self.bg_sigma, sigmaY=self.bg_sigma)


        if self.deeper_is_lower_z:
            Rz = np.maximum(Z_bg - Z_f, 0.0)
        else:
            Rz = np.maximum(Z_f - Z_bg, 0.0)


        Rz = Rz * vis_mask.astype(np.float32)
        Rz = np.clip(Rz, 0.0, self.r_max)
        H = Rz / self.r_max


        # ---- forward-first guidance line ----
        rows = np.linspace(v1 - 1, v0, self.num_guidance_layers).astype(np.int32)
        rows = np.unique(rows)[::-1]
        if len(rows) < 2:
            return H, Z_level, vis_mask, ([], []), 0.0


        image_center = 0.5 * (u0 + u1)
        if self.prev_u_ref is None:
            u_ref = image_center
        else:
            u_ref = self.prev_u_ref


        u_list: List[int] = []
        v_list: List[int] = []
        chosen_scores: List[float] = []


        for vrow in rows:
            row_heat = H[vrow, :]
            row_mask = vis_mask[vrow, :] > 0


            # 当前层局部搜索窗口
            left = max(u0, int(round(u_ref - self.search_half_width)))
            right = min(u1 - 1, int(round(u_ref + self.search_half_width)))


            if right <= left:
                continue


            cols = np.arange(left, right + 1, dtype=np.int32)
            valid_cols = row_mask[cols]


            if not np.any(valid_cols):
                continue


            cols_valid = cols[valid_cols]
            heat_valid = row_heat[cols_valid]


            # 正前方候选：离 u_ref 最近的合法列
            idx_forward = int(np.argmin(np.abs(cols_valid.astype(np.float64) - u_ref)))
            u_forward = int(cols_valid[idx_forward])
            h_forward = float(heat_valid[idx_forward])


            # 局部最优候选
            idx_best = int(np.argmax(heat_valid))
            u_best = int(cols_valid[idx_best])
            h_best = float(heat_valid[idx_best])


            # 如果正前方足够好，则优先保持前进
            if h_best > 1e-6 and h_forward >= self.forward_keep_margin * h_best:
                u_choose = u_forward
                h_choose = h_forward
            else:
                # 否则在局部窗口内做“低点优先 + 偏移惩罚 + 前向偏置”
                deviation = np.abs(cols_valid.astype(np.float64) - u_ref)
                step_penalty = self.deviation_weight * deviation


                # 正前方偏置：离 u_ref 越近越鼓励
                forward_bonus = self.forward_bias_weight * (
                    1.0 - np.clip(deviation / max(1.0, self.search_half_width), 0.0, 1.0)
                )


                score = heat_valid + forward_bonus - step_penalty
                idx_choose = int(np.argmax(score))
                u_choose = int(cols_valid[idx_choose])
                h_choose = float(heat_valid[idx_choose])


            # 限制单层横跳
            du = u_choose - u_ref
            du = np.clip(du, -self.max_step_u, self.max_step_u)
            u_choose = int(round(u_ref + du))


            u_choose = max(u0, min(u1 - 1, u_choose))


            u_list.append(u_choose)
            v_list.append(int(vrow))
            chosen_scores.append(h_choose)


            u_ref = float(u_choose)


        if len(u_list) >= 2:
            self.prev_u_ref = float(u_list[min(1, len(u_list) - 1)])
        else:
            self.prev_u_ref = float(u_ref)


        conf = float(np.mean(chosen_scores)) if len(chosen_scores) > 0 else 0.0
        return H, Z_level, vis_mask, (u_list, v_list), conf


    # ---------------- adaptive visualization ----------------
    def make_adaptive_heat_color(self, H: np.ndarray, vis_mask: np.ndarray) -> np.ndarray:
        valid = (vis_mask > 0) & np.isfinite(H) & (H > 0.0)
        if np.count_nonzero(valid) < 20:
            heat_u8 = (np.clip(H, 0.0, 1.0) * 255.0).astype(np.uint8)
            out = cv2.applyColorMap(heat_u8, self.colormap)
            out[vis_mask == 0] = 0
            return out


        vals = H[valid]
        lo = np.percentile(vals, self.vis_percentile_low)
        hi = np.percentile(vals, self.vis_percentile_high)


        if hi <= lo + 1e-6:
            heat_u8 = (np.clip(H, 0.0, 1.0) * 255.0).astype(np.uint8)
            out = cv2.applyColorMap(heat_u8, self.colormap)
            out[vis_mask == 0] = 0
            return out


        Hn = np.zeros_like(H, dtype=np.float32)
        Hn[valid] = (H[valid] - lo) / (hi - lo)
        Hn = np.clip(Hn, 0.0, 1.0)
        Hn = np.power(Hn, 0.7)


        heat_u8 = (Hn * 255.0).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, self.colormap)
        heat_color[vis_mask == 0] = 0
        return heat_color


    # ---------------- 3D in camera_level_frame ----------------
    def uv_to_3d_level(
        self,
        D: np.ndarray,
        u_list: List[int],
        v_list: List[int],
        K: CamK,
        R: np.ndarray,
        t: np.ndarray,
        step: int = 1
    ) -> List[Tuple[float, float, float]]:
        pts_out = []
        for i in range(0, len(v_list), step):
            u = u_list[i]
            v = v_list[i]
            z = float(D[v, u])
            if not np.isfinite(z) or z <= 0.0:
                continue


            x = (u - K.cx) * z / K.fx
            y = (v - K.cy) * z / K.fy
            p = np.array([x, y, z], dtype=np.float64)
            p2 = R @ p + t
            pts_out.append((float(p2[0]), float(p2[1]), float(p2[2])))
        return pts_out

    # ---------------- Path / Marker ----------------
    # def make_path(self, header, pts: List[Tuple[float, float, float]]) -> Path:
    #     path = Path()
    #     path.header = header
    #     for (x, y, z) in pts:
    #         ps = PoseStamped()
    #         ps.header = header
    #         ps.pose.position.x = float(x)
    #         ps.pose.position.y = float(y)
    #         ps.pose.position.z = float(z)
    #         ps.pose.orientation.w = 1.0
    #         path.poses.append(ps)
    #     return path

    # ---------------- Path / Marker ----------------
    def make_path(self, header, pts: List[Tuple[float, float, float]]) -> Path:
        path = Path()
        path.header = header
        # 【修改】不再强制覆盖 frame_id，直接使用外面传进来的 header.frame_id
        
        if not pts:
            return path
            
        n_pts = len(pts)
        arr = np.array(pts, dtype=np.float64)
        
        # 计算轨迹的切线方向
        yaws = np.zeros(n_pts, dtype=np.float64)
        if n_pts >= 2:
            dx = np.diff(arr[:, 0])
            dy = np.diff(arr[:, 1])
            segment_yaws = np.arctan2(dy, dx)
            yaws[:-1] = segment_yaws
            yaws[-1] = segment_yaws[-1]
            if n_pts >= 3:
                ds = np.hypot(dx, dy)
                s = np.concatenate(([0.0], np.cumsum(ds)))
                smooth_dx = np.gradient(arr[:, 0], s, edge_order=2)
                smooth_dy = np.gradient(arr[:, 1], s, edge_order=2)
                yaws = np.unwrap(np.arctan2(smooth_dy, smooth_dx))
        
        # 【修改】直接使用传入的点，不做任何偏移，因为外面已经转换好了
        for i in range(n_pts):
            px, py, pz = pts[i]
            pyaw = yaws[i]
            
            # 转换为四元数
            qz = math.sin(pyaw / 2.0)
            qw = math.cos(pyaw / 2.0)
            
            # 组装消息
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.position.z = float(pz)
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = float(qz)
            ps.pose.orientation.w = float(qw)
            
            path.poses.append(ps)
            
        return path


    def make_marker(self, header, pts: List[Tuple[float, float, float]]) -> Marker:
        mk = Marker()
        mk.header = header
        mk.ns = "furrow"
        mk.id = 0
        mk.type = Marker.LINE_STRIP
        mk.action = Marker.ADD
        mk.scale.x = 0.05
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


    def make_target_marker(self, header, target_pt: np.ndarray) -> Marker:
        mk = Marker()
        mk.header = header
        mk.ns = "furrow_target"
        mk.id = 1
        mk.type = Marker.SPHERE
        mk.action = Marker.ADD
        mk.scale.x = 0.05
        mk.scale.y = 0.05
        mk.scale.z = 0.05
        mk.color.a = 1.0
        mk.color.r = 0.1
        mk.color.g = 1.0
        mk.color.b = 0.1


        mk.pose.position.x = float(target_pt[0])
        mk.pose.position.y = float(target_pt[1])
        mk.pose.position.z = float(target_pt[2])
        mk.pose.orientation.w = 1.0
        return mk


    # ---------------- overlay ----------------
    def make_overlay(self, color_bgr: np.ndarray, heat_color_bgr: np.ndarray, H: np.ndarray, vis_mask: np.ndarray) -> np.ndarray:
        hC, wC = color_bgr.shape[:2]
        hH, wH = heat_color_bgr.shape[:2]


        if (hC != hH) or (wC != wH):
            heat_rs = cv2.resize(heat_color_bgr, (wC, hC), interpolation=cv2.INTER_NEAREST)
            H_rs = cv2.resize(H, (wC, hC), interpolation=cv2.INTER_NEAREST)
            mask_rs = cv2.resize(vis_mask, (wC, hC), interpolation=cv2.INTER_NEAREST)
        else:
            heat_rs = heat_color_bgr
            H_rs = H
            mask_rs = vis_mask


        overlay = color_bgr.copy()


        alpha_map = np.zeros_like(H_rs, dtype=np.float32)
        valid = (mask_rs > 0) & (H_rs >= self.vis_alpha_threshold)
        alpha_map[valid] = self.overlay_alpha * np.clip(H_rs[valid], 0.0, 1.0)


        for c in range(3):
            overlay[..., c] = (
                (1.0 - alpha_map) * overlay[..., c].astype(np.float32)
                + alpha_map * heat_rs[..., c].astype(np.float32)
            ).astype(np.uint8)


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