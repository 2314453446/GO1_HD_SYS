# aarch64_l3_envsdk_requirements.dockerfile
# L3: Algo / SDK deps on top of ROS2+Orbbec (L2)

FROM openorbbecsdk-env:aarch64_ros2_orbbec_latest

ENV DEBIAN_FRONTEND=noninteractive
# 关键：避免容器读到宿主机 ~/.local（你之前的 .pth 污染问题）
ENV PYTHONNOUSERSITE=1

# ---- OS deps (keep minimal; base already has many) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    python3-opencv \
    libjpeg-dev \
    libpng-dev \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# ---- pip baseline ----
RUN python3 -m pip install -U --no-cache-dir pip setuptools wheel

# ---- IMPORTANT: pin numpy < 2 to keep cv2/other binary extensions working ----
# 推荐 1.26.4：兼容面最广（你之前 numpy 2.x 已经把 cv2 搞崩了）
RUN python3 -m pip install --no-cache-dir "numpy==1.26.4"

# ---- common python deps (add what your YOLO/ROS scripts need) ----
RUN python3 -m pip install --no-cache-dir \
    pandas \
    scipy \
    matplotlib \
    tqdm \
    pillow \
    h5py \
    python-dateutil \
    tzdata

# ---- Orbbec python SDK (choose one; keep as optional if your project needs it) ----
# 如果你实际用的是 pyorbbecsdk / pyorbbecsdk2，保留其中一个即可
#RUN python3 -m pip install --no-cache-dir pyorbbecsdk || true
RUN python3 -m pip install --no-cache-dir pyorbbecsdk2 || true

# ---- Torch on Jetson: DO NOT pip install "torch" from PyPI blindly ----
# If you have a Jetson/NVIDIA wheel file inside build context, you can enable this:
#   docker build --build-arg TORCH_WHEEL=torch-2.x.x+nvXX.whl ...
# ---- Torch family (Jetson offline wheels) ----
COPY wheels/ /tmp/wheels/

RUN python3 -m pip install --no-cache-dir \
        /tmp/wheels/torch-2.3.0-cp310-cp310-linux_aarch64.whl \
        /tmp/wheels/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl \
        /tmp/wheels/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl \
    && rm -rf /tmp/wheels


WORKDIR /ws
