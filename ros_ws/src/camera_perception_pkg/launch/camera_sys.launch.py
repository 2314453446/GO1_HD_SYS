#对gemini_330_series.launch.py进行二次封装，读取校正参数camera_level.yaml,发布与地面平行的camrea_level_frame

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource

import os
import subprocess


def generate_launch_description():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 你的 TF 发布脚本（和 launch 放同一目录）
    tf_pub_script = os.path.join(current_dir, "publish_camera_level_tf.py")

    # 查找 orbbec_camera 官方 launch
    pkg_prefix = subprocess.check_output(
        ["ros2", "pkg", "prefix", "orbbec_camera"]
    ).decode().strip()

    orbbec_launch_file = os.path.join(
        pkg_prefix,
        "share",
        "orbbec_camera",
        "launch",
        "gemini_330_series.launch.py"
    )

    orbbec_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(orbbec_launch_file),
        launch_arguments={
            "enable_colored_point_cloud": "true",
            "enable_accel": "true",
            "enable_gyro": "true",
            "enable_sync_output_accel_gyro": "true",
        }.items()
    )

    # 运行当前目录下的 TF 发布脚本
    camera_level_tf = ExecuteProcess(
        cmd=["python3", tf_pub_script],
        output="screen"
    )

    return LaunchDescription([
        orbbec_launch,
        camera_level_tf,
    ])