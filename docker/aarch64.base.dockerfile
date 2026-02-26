# aarch64.base.dockerfile
# Jetson R36 / JetPack 6.x / Ubuntu 22.04 base

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# Optional: 国内源加速（只改 Ubuntu ports 源）
RUN sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.aliyun.com/ubuntu-ports|g' /etc/apt/sources.list

ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

ARG NVIDIA_VISIBLE_DEVICES=all
ARG NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video

# 通用构建/运行依赖（后续所有派生镜像复用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    git \
    curl \
    ca-certificates \
    pkg-config \
    zip \
    udev \
    tzdata \
    openssl \
    libssl-dev \
    libusb-1.0-0-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libnuma1 \
    python3 \
    python3-dev \
    python3-pip \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# CMake：22.04 自带满足 >=3.15；如你希望固定到 3.30，可以保留
RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0-linux-aarch64.tar.gz -o cmake.tar.gz \
    && tar -xzf cmake.tar.gz --strip-components=1 -C /usr/local \
    && rm cmake.tar.gz

# NVIDIA EGL / Vulkan（保留你原始结构）
ADD 10_nvidia.json /etc/glvnd/egl_vendor.d/10_nvidia.json
RUN chmod 644 /etc/glvnd/egl_vendor.d/10_nvidia.json

ADD nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
RUN chmod 644 /etc/vulkan/icd.d/nvidia_icd.json

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}

# ADE / Entry（保持你的工作流）
COPY env.sh /etc/profile.d/ade_env.sh
COPY entrypoint.sh /ade_entrypoint
RUN chmod +x /ade_entrypoint

ENTRYPOINT ["/ade_entrypoint"]
CMD ["/bin/bash"]
