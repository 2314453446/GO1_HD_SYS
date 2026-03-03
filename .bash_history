exit
docker ps
exit
lsb_release -a
lsb_release -r
cat /etc/issue
exit
exit 
cd 
ll
cd dev/
ll
cd ..
ll
cd third_party/
ll
git clone https://github.com/orbbec/pyorbbecsdk.git
chmod +x ./install_udev_rules.sh
./install_udev_rules.sh
sudo ./install_udev_rules.sh
sudo -i
exit
sudo
ll
cd third_party/
docker exec -u 0 -it camera bash
# 如果没有 bash 就用：
# docker exec -u 0 -it camera sh
id unitreego1
usermod -aG sudo unitreego1
su root
exit
exot
exit
ll
cd third_party/
ll
cd pyorbbecsdk/
cd scripts/
chmod +x ./install_udev_rules.sh
./install_udev_rules.sh
sudo ./install_udev_rules.sh
su root
sudo -i
exit
cd ~/third_party/pyorbbecsdk/scripts
sudo chmod +x ./install_udev_rules.sh
sudo ./install_udev_rules.sh
sudo udevadm control --reload && sudo udevadm trigger
ll
./install_udev_rules.sh 
sudo ./install_udev_rules.sh 
pip3 install pyorbbecsdk2
python3 --version
cd
cd third_party/
ll
python3 -m pip install pyorbbecsdk-2.0.15-cp310-cp310-linux_aarch64.whl 
python3 -m pip show pyorbbecsdk
python3 -m pip install -r /home/unitreego1/.local/lib/python3.10/site-packages/pyorbbecsdk/examples/requirements.txt
ll
cd pyorbbecsdk
python examples/color.py 
exit
echo "DISPLAY=$DISPLAY"
ls -la /tmp/.X11-unix
exit
cd third_party/pyorbbecsdk
cd examples/
python point_cloud.py 
cd ~/third_party/
git clone https://github.com/bubbliiiing/yolov7-pytorch.git
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
zhangm
which pip
python3 -m pip install yolov7_pytorch/requirements-py310-compat.txt 
python3 -m pip install -r yolov7_pytorch/requirements-py310-compat.txt 
python yolov7_pytorch/predict
python yolov7_pytorch/predict.py 
python3 -m pip install pandas
python yolov7_pytorch/predict.py 
cd yolov7_pytorch/
python yolov7_pytorch/predict.py 
python predict.py 
free -h
python predict.py 
python3 - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda in torch:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("arch_list:", torch.cuda.get_arch_list())
PY

python3 -m pip uninstall -y torch torchvision torchaudio
python3 -m pip cache purge
python3 -m pip install --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch torchvision
python predict.py 
sudo apt-get update
sudo apt-get install -y libopenblas0 libopenblas-dev
sudo ldconfig
apt-get update
su root
sudo -t
sudo -i
exit
cd third_party/yolov7_pytorch/
python predict.py 
python3 - <<'PY'
import torch, os
import torch._C as C
print("torch ok:", torch.__version__)
print("torch._C:", C.__file__)
PY

sudo apt-get update
sudo apt-get install -y libopenblas0 libopenblas-dev
sudo ldconfig
exit
cd third_party/yolov7_pytorch/
python predict.py 
sudo apt-get update
sudo apt-get install -y libopenblas0 libopenblas-dev
sudo ldconfig
exit
cd third_party/yolov7_pytorch/
sudo apt-get update
sudo apt-get install -y libopenblas0 libopenblas-dev
sudo ldconfig
su root
sudo passwd unitreego1
passwd unitreego1
sudo apt-get update
sudo apt-get install -y libopenblas0 libopenblas-dev
sudo ldconfig
sudo passwd unitreego1
su root
apt-get update
apt-get install -y libopenblas0 libopenblas-dev
ldconfig
sudo apt-get update
sudo apt-get install -y libopenblas0 libopenblas-dev
sudo ldconfig
su root
sudo -v
exit
sudo -v
su root
exit
sudo apt-get update
sudo apt-get install -y libopenblas0 libopenblas-dev
sudo ldconfig
python3 -c "import torch; print(torch.__version__)"
su root
exit
sudo apt-get update
sudo apt-get install -y libopenblas0 libopenblas-dev
sudo ldconfig
python3 -c "import torch; print(torch.__version__)"
su root
sudo -v
passwd ubitreego1
passwd unitreego1
sudo passwd unitreego1
exit
ll
su roo
su root
ll
cd /
ll
sudo vim ade_entrypoint 
vim ade_entrypoint 
sudo apt intstall vim
apt-get install -y vim
sudo apt-get install -y vim
sudo apt-get update
sudo apt-get install -y vim
sudo vim /ade_entrypoint 
exit
sudo -v
# 1) 容器基础信息
cat /etc/os-release | head
python3 --version
uname -m
# 2) GPU/driver 是否进容器
ls -l /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so* 2>/dev/null | head || echo "no libcuda in container"
ls -l /dev/nv* 2>/dev/null | head || echo "no /dev/nv* in container"
# 3) 当前是否已经装了 torch
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)" 2>/dev/null || echo "torch not working"
python3 -m pip show torch | sed -n '1,25p' || true
cat /etc/nv_tegra_release 2>/dev/null || true
python -c "import sys; print(sys.version)"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("avail:", torch.cuda.is_available())
PY

sudo apt-get update
sudo apt-get install -y git build-essential cmake ninja-build   libjpeg-dev zlib1g-dev libpng-dev
# 确保用的就是当前这个 python/pip（别用 pip3 乱指）
python -m pip install -U pip setuptools wheel
git clone --branch v0.20.1 --depth 1 https://github.com/pytorch/vision /tmp/torchvision
cd /tmp/torchvision
export FORCE_CUDA=1
python -m pip install -v --no-deps .
python3 - <<'PY'
import setuptools, pkg_resources
print("setuptools:", setuptools.__version__)
print("setuptools file:", setuptools.__file__)
print("pkg_resources OK")
PY

python3 -m pip uninstall -y setuptools
rm -rf /tmp/torchvision /tmp/pip-build-env-* /tmp/pip-*
rm -rf ~/.cache/pip
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
PY

cd ~
pwd
python3 -m pip --version
python3 -m pip list --user | head -n 50
python3 - <<'PY'
import ctypes
nvml = ctypes.CDLL("libnvidia-ml.so.1")
nvmlInit_v2 = nvml.nvmlInit_v2
nvmlInit_v2.restype = ctypes.c_int
ret = nvmlInit_v2()
print("nvmlInit_v2 ret =", ret)
PY

cd third_party/yolov7_pytorch/
python ./predict.py 
free -h
python3 - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("mem_get_info:", torch.cuda.mem_get_info())  # (free, total)
PY

python ./predict.py 
python3 - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("mem_get_info:", torch.cuda.mem_get_info())  # (free, total)
PY

ps -eo pid,user,comm,rss --sort=-rss | head -n 20
python ./predict.py 
exit
cd third_party/yolov7_pytorch/
python ./predict.py 
python - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH",""))
PY

python - <<'PY'
import torch
print("cuda:", torch.version.cuda, "cudnn:", torch.backends.cudnn.version())
x = torch.randn(1,3,640,640, device='cuda')
w = torch.randn(32,3,3,3, device='cuda')
y = torch.nn.functional.conv2d(x, w, None, 1, 1)
print("ok", y.shape)
PY

python - <<'PY'
import torch
torch.backends.cudnn.enabled = False
x = torch.randn(1,3,640,640, device='cuda')
w = torch.randn(32,3,3,3, device='cuda')
y = torch.nn.functional.conv2d(x, w, None, 1, 1)
print("ok(no cudnn)", y.shape)
PY

python - <<'PY'
import torch, torch._C
print(torch._C.__file__)
PY

python -m pip uninstall -y   nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12   nvidia-cuda-nvrtc-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12   || true
apt-cache search libcudnn | head
dpkg -l | grep -i cudnn || true
sudo apt-get update
# 下面这行用你系统里实际存在的包名替换（常见是 libcudnn9 / libcudnn9-dev）
sudo apt-get install --reinstall -y libcudnn9 libcudnn9-dev
sudo ldconfig
python - <<'PY'
import torch, torch._C
print(torch._C.__file__)
PY

ldd $(python - <<'PY'
import torch, torch._C
print(torch._C.__file__)
PY
) | egrep "cudnn|cublas|cudart|cuda|nvrtc" | sort
readlink -f /lib/aarch64-linux-gnu/libcudnn.so.9
ls -l /lib/aarch64-linux-gnu/libcudnn.so.9*
strings /lib/aarch64-linux-gnu/libcudnn.so.9 | grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL|cudnn_version" | head -n 20 || true
ldconfig -p | grep -E "libcudnn\.so\.9" -n
sudo apt-get install --reinstall -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
sudo ldconfig
readlink -f /lib/aarch64-linux-gnu/libcudnn.so.9
ls -l /lib/aarch64-linux-gnu/libcudnn.so.9*
ldconfig -p | grep libcudnn.so.9
python ./predict.py 
ls -l /lib/aarch64-linux-gnu/libcudnn* /usr/lib/aarch64-linux-gnu/libcudnn* 2>/dev/null | head -n 200
dpkg -S /lib/aarch64-linux-gnu/libcudnn.so.9.19.0
dpkg -l | grep -E "libcudnn9|nvidia-cudnn9" || true
apt-cache policy libcudnn9-cuda-12 libcudnn9-dev-cuda-12 | sed -n '1,160p'
sudo apt-get purge -y "libcudnn9*" "nvidia-cudnn9*" || true
sudo apt-get autoremove -y
sudo ldconfig
sudo apt-get update
sudo apt-get install -y --reinstall   libcudnn9-cuda-12   libcudnn9-dev-cuda-12   libcudnn9-headers-cuda-12   libcudnn9-samples
sudo ldconfig
python ./predict.py 
python - <<'PY'
import torch
print(torch.backends.cudnn.version())
PY

ls -l /lib/aarch64-linux-gnu/libcudnn* /usr/lib/aarch64-linux-gnu/libcudnn* 2>/dev/null | head -n 200
python - <<'PY'
import torch
print(torch.backends.cudnn.version())
PY

python predict.py 
which xdg-open || echo "no xdg-open"
cd /tmp/
ll
ego tmpfvg42ksw.PNG 
feh tmpfvg42ksw.PNG 
file tmpfvg42ksw.PNG
ls -lh tmpfvg42ksw.PNG
python3 - <<'PY'
import struct
p="tmpfvg42ksw.PNG"
with open(p,'rb') as f:
    sig=f.read(8)
    print("sig",sig)
    if sig != b"\x89PNG\r\n\x1a\n":
        print("NOT a PNG signature")
    # 读 IHDR chunk
    ln=struct.unpack(">I", f.read(4))[0]
    typ=f.read(4)
    data=f.read(ln)
    print("chunk", typ, "len", ln)
    if typ==b'IHDR' and ln==13:
        w,h,bd,ct,cm,fl,il = struct.unpack(">IIBBBBB", data)
        print("W H bitdepth colortype:", w,h,bd,ct)
PY

pngcheck -v tmpfvg42ksw.PNG | head -n 80
sudo apt-get update
sudo apt-get install -y pngcheck
pngcheck -v tmpfvg42ksw.PNG | head -n 120
feh tmpfvg42ksw.PNG 
cd
ll
cd third_party/yolov7_pytorch/
pwd
cd  
ll
cd ros_ws/
ll
cd src/camera_perception_pkg/
python predict_weed.py 
[A
python predict_weed.py 
exit
cd ros_ws/src/camera_perception_pkg/
python predict_weed.py 
which python3
python3 -m site
python3 -m pip install pandas
sudo python predict_weed.py 
sudo python3 -m pip install pandas
python3 predict_weed.py 
cd ~/third_party/pyorbbecsdk/examples/
python color.py 
exit
cd third_party/yolov7_pytorch/
cd ~/ros_ws/
Ll
ll
cd src/
ll
cd camera_perception_pkg/
python predict_weed.py 
exit
cd ~/ros_ws/src/camera_perception_pkg/
python predict_weed.py 
docker images
exit
which python
pip list
which python3
pip list
exit
cd ~/ros_ws/src/camera_perception_pkg/
python predict_weed.py 
exit
cd ros_ws/src/camera_perception_pkg/
python predict_weed.py 
exit
cd ros_ws/src/camera_perception_pkg/
python predict_weed.py \
python predict_weed.py 
python3 -c "import pyorbbecsdk"
pip3 list | grep -i orbbec
exit
