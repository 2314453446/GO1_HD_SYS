from pathlib import Path
import os
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
# 新增：同步工具
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import threading
from sensor_msgs.msg import CameraInfo

# A) 把“当前脚本所在目录”加到 sys.path（保证能 import utils.py）
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# 1) 设置运行根目录（yolov7 根）
ROOT = Path("/home/unitreego1/third_party/yolov7_pytorch").resolve()
if not ROOT.exists():
    raise FileNotFoundError(f"ROOT not found: {ROOT}")
# 切换工作目录（影响相对路径）
os.chdir(ROOT)
# （强烈建议）同时加入 import 搜索路径
sys.path.insert(0, str(ROOT))

print("cwd =", Path.cwd())


import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO, YOLO_ONNX


class RosOrbbecYolo(Node):
    def __init__(self, yolo, video_save_path="", video_fps=25, count=False):
        super().__init__('yolo_ros_node')

        self.bridge = CvBridge()
        self.yolo = yolo
        self.count = count
        self.out = None
        self.fps = 0.0

        self.video_save_path = video_save_path
        self.video_fps = video_fps

        # camera_info 缓存（color 内参，因 depth 已 registration 到 color）
        self._info_lock = threading.Lock()
        self._color_info = None  # sensor_msgs/CameraInfo

        # ⭐ 模型只处理一次（沿用你原逻辑）
        if hasattr(self.yolo, "net"):
            self.yolo.net.eval()
            if getattr(self.yolo, "cuda", False):
                try:
                    self.yolo.net = self.yolo.net.cuda()
                except Exception:
                    pass
                try:
                    self.yolo.net.half()
                    print("[YOLO] FP16 enabled")
                except Exception as e:
                    print("[YOLO] FP16 fallback:", e)

        # ⭐⭐ 订阅 camera_info（color）
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.info_callback,
            10
        )

        # ⭐⭐⭐ RGB + Depth 同步订阅
        self.rgb_sub = Subscriber(self, ROSImage, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, ROSImage, '/camera/depth/image_raw')

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.05
        )
        self.sync.registerCallback(self.callback)

        self.get_logger().info("ROS YOLO node started (RGB+Depth synced + CameraInfo cached).")

    def info_callback(self, msg: CameraInfo):
        with self._info_lock:
            self._color_info = msg

    def _get_k(self):
        """
        取相机内参 (fx, fy, cx0, cy0)。取不到返回 None。
        """
        with self._info_lock:
            info = self._color_info

        if info is None:
            return None

        fx = float(info.k[0])
        fy = float(info.k[4])
        cx0 = float(info.k[2])
        cy0 = float(info.k[5])

        if fx <= 0 or fy <= 0:
            return None

        return fx, fy, cx0, cy0

    def uv_list_to_xyz_list(self, uv_list, depth_m, use_median=True, win=1):
        """
        把一组像素点 (u,v) 转换成相机坐标系 (x,y,z)（单位 m）。
        - uv_list: [(u,v), ...]
        - depth_m: HxW 的深度图（单位 m），且与 RGB 已 registration 对齐
        - use_median: 是否在 (u,v) 周围取中位数深度（更稳）
        - win: 半窗口大小；win=1 => 3x3；win=2 => 5x5

        返回：
        - xyz_list: 与 uv_list 等长，元素为 (x,y,z) 或 None
        """
        k = self._get_k()
        if k is None:
            return [None] * len(uv_list)

        fx, fy, cx0, cy0 = k
        H, W = depth_m.shape[:2]

        def get_depth(u, v):
            if u < 0 or u >= W or v < 0 or v >= H:
                return float('nan')

            if not use_median:
                return float(depth_m[v, u])

            u0 = max(0, u - win)
            u1 = min(W, u + win + 1)
            v0 = max(0, v - win)
            v1 = min(H, v + win + 1)

            patch = depth_m[v0:v1, u0:u1].astype(np.float32)
            patch = patch[np.isfinite(patch)]
            patch = patch[patch > 0]

            if patch.size == 0:
                return float('nan')

            return float(np.median(patch))

        xyz_list = []

        for (u, v) in uv_list:
            u = int(u)
            v = int(v)

            z = get_depth(u, v)

            if (not np.isfinite(z)) or z <= 0:
                xyz_list.append(None)
                continue

            x = (u - cx0) * z / fx
            y = (v - cy0) * z / fy
            xyz_list.append((float(x), float(y), float(z)))

        return xyz_list

    def callback(self, rgb_msg, depth_msg):
        t1 = time.time()

        # ===== ROS → OpenCV (RGB) =====
        frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        if frame is None:
            return

        # ===== ROS → OpenCV (Depth) =====
        if getattr(depth_msg, "encoding", "") in ('32FC1', '32FC'):
            depth_m = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')  # meters
        elif getattr(depth_msg, "encoding", "") in ('16UC1', 'mono16'):
            depth_u16 = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')  # millimeters
            depth_m = depth_u16.astype(np.float32) / 1000.0
        else:
            depth_any = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
            depth_m = depth_any.astype(np.float32)

        if depth_m is None:
            return

        # ===== BGR→RGB→PIL（保持你原流程）=====
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # 需要你在 yolo.py 里实现 return_uv 参数：
        #   return_uv=False -> return image
        #   return_uv=True  -> return image, uv_list
        pil_out, uv_list = self.yolo.detect_orbbec_image(pil, count=self.count, return_uv=True)
        frame = cv2.cvtColor(np.asarray(pil_out), cv2.COLOR_RGB2BGR)

        # ===== 把 YOLO 检测到的 uv_list 映射成 xyz_list =====
        xyz_list = self.uv_list_to_xyz_list(uv_list, depth_m, use_median=True, win=1)

        # ===== 可视化：画中心点 + 标注 3D 坐标（最多显示前 N 个，避免画面太乱）=====
        max_show = 10
        for i, (uv, xyz) in enumerate(zip(uv_list, xyz_list)):
            if i >= max_show:
                break

            u, v = int(uv[0]), int(uv[1])

            # 越界保护
            H, W = depth_m.shape[:2]
            if u < 0 or u >= W or v < 0 or v >= H:
                continue

            cv2.circle(frame, (u, v), 4, (0, 0, 255), -1)

            if xyz is None:
                text = "xyz=nan"
            else:
                x, y, z = xyz
                text = f"({x:.2f},{y:.2f},{z:.2f})m"

            # 文本位置稍微偏移，避免压在点上
            tx = min(u + 6, frame.shape[1] - 1)
            ty = max(v - 6, 0)

            cv2.putText(
                frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        # ===== FPS =====
        self.fps = (self.fps + (1.0 / (time.time() - t1))) / 2.0
        print("fps= %.2f" % self.fps)

        frame = cv2.putText(
            frame, f"fps= {self.fps:.2f}", (0, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # ===== 显示 =====
        cv2.imshow("video", frame)

        depth_vis = depth_m.copy()
        maxv = np.nanmax(depth_vis)
        if maxv > 0:
            depth_vis = depth_vis / maxv
        cv2.imshow("depth", depth_vis)

        cv2.waitKey(1)

        # ===== 视频保存 =====
        if self.video_save_path != "":
            if self.out is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                size = (frame.shape[1], frame.shape[0])
                self.out = cv2.VideoWriter(
                    self.video_save_path, fourcc, self.video_fps, size
                )
            self.out.write(frame)

if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'orbbec_video'      表示orbbec 336相机实时输入
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #   'predict_onnx'      表示利用导出的onnx模型进行预测，相关参数的修改在yolo.py_423行左右处的YOLO_ONNX
    # ----------------------------------------------------------------------------------------------------------#
    mode = "ros_image"
    # -------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    crop = False
    count = False
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = r'D:\masterPROJECT\paper\object_detection\figures\result_video\origin_video.avi'
    video_save_path = r'D:\masterPROJECT\paper\object_detection\figures\result_video\yolo_video.avi'
    video_fps = 5.0
    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #
    #   test_interval和fps_image_path仅在mode='fps'有效
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    # -------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #
    #   heatmap_save_path仅在mode='heatmap'有效
    # -------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision"
    # -------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode != "predict_onnx":
        yolo = YOLO()
    else:
        yolo = YOLO_ONNX()

    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop=crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame, count=True))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()






    elif mode == "ros_image":

        rclpy.init()

        node = RosOrbbecYolo(

            yolo,

            video_save_path=video_save_path,

            video_fps=video_fps,

            count=count

        )

        try:

            rclpy.spin(node)


        finally:

            if node.out is not None:
                node.out.release()

            node.destroy_node()

            rclpy.shutdown()

            cv2.destroyAllWindows()



    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap_eachlayer_v2(image, heatmap_save_path)

    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError(
            "Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
