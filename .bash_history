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
