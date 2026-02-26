# aarch64.orbbec_ros2_driver.dockerfile
# Derived from your base image: add ROS2 Humble (desktop) + build Orbbec ROS2 wrapper

ARG BASE_IMAGE=openorbbecsdk-env:aarch64_base
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DISTRO=humble

# ---- ROS2 apt repo (jammy) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/keyrings \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list

# ---- ROS2 Humble desktop + build tools ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true && rosdep update || true

# ---- Workspace ----
ARG WS=/ros2_ws
WORKDIR ${WS}
RUN mkdir -p ${WS}/src

# ---- Orbbec ROS2 wrapper ----
# OrbbecSDK_ROS2 supports Humble, default branch v2-main (can override)  (see repo README)
ARG ORBBEC_ROS2_REPO=https://github.com/orbbec/OrbbecSDK_ROS2.git
ARG ORBBEC_ROS2_BRANCH=v2-main

WORKDIR ${WS}/src
RUN git clone --depth 1 -b ${ORBBEC_ROS2_BRANCH} ${ORBBEC_ROS2_REPO}

# ---- Resolve deps + build ----
WORKDIR ${WS}
RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
    && apt-get update \
    && rosdep install -y --from-paths src --ignore-src -r --rosdistro ${ROS_DISTRO} \
    && rm -rf /var/lib/apt/lists/* \
    && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# ---- Auto-source ROS + overlay ----
RUN printf '%s\n' \
 'if [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then source "/opt/ros/$ROS_DISTRO/setup.bash"; fi' \
 'if [ -f "/ros2_ws/install/setup.bash" ]; then source "/ros2_ws/install/setup.bash"; fi' \
 > /etc/profile.d/ros2_setup.sh \
 && chmod +x /etc/profile.d/ros2_setup.sh \
 && echo "source /etc/profile.d/ros2_setup.sh" >> /etc/bash.bashrc

# Keep ADE workflow entrypoint
ENTRYPOINT ["/ade_entrypoint"]
CMD ["/bin/bash", "-c", "trap 'exit 147' TERM; tail -f /dev/null & while wait ${!}; test $? -ge 128; do true; done"]
