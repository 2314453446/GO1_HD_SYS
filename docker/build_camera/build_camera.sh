#!/usr/bin/env bash
set -euo pipefail

# === 关键：自动定位脚本所在目录，以及项目根目录===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"   # 
CONTEXT="${REPO_ROOT}"

echo "Please select the architecture:"
echo "1) x86_64"
echo "2) aarch64"
read -p "Enter your choice (1 or 2): " choice

case $choice in
  1) ARCH="x86_64" ;;
  2) ARCH="aarch64" ;;
  *) echo "Invalid choice"; exit 1 ;;
esac

DATE=$(date +'%Y%m%d')
TAG_TIME=$(date +'%Y%m%d.%H%M%S')

# git 信息也要在 REPO_ROOT 下取，避免你在 build_script 下运行时报 nogit
if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
else
  GIT_SHA="nogit"
fi

# Dockerfile（用绝对路径最稳）
BASE_DOCKERFILE="${REPO_ROOT}/${ARCH}.base.dockerfile"
DERIVED_DOCKERFILE="${REPO_ROOT}/${ARCH}.orbbec_ros2_driver.dockerfile"

# Image tags
BASE_TAG_DATE="openorbbecsdk-env:${ARCH}_base_${DATE}"
BASE_TAG_STABLE="openorbbecsdk-env:${ARCH}_base"

DERIVED_TAG_DATE="openorbbecsdk-env:${ARCH}_ros2_orbbec_${DATE}"
DERIVED_TAG_STABLE="openorbbecsdk-env:${ARCH}_ros2_orbbec_latest"

# ---- build base ----
if [ ! -f "${BASE_DOCKERFILE}" ]; then
  echo "ERROR: ${BASE_DOCKERFILE} not found."
  exit 1
fi

echo "[1/2] Build base image: ${BASE_TAG_DATE}"
docker build \
  -f "${BASE_DOCKERFILE}" \
  -t "${BASE_TAG_DATE}" \
  --label ade_image_commit_sha="${GIT_SHA}" \
  --label ade_image_commit_tag="${TAG_TIME}" \
  "${CONTEXT}"

# stable tag for chaining
docker tag "${BASE_TAG_DATE}" "${BASE_TAG_STABLE}"
echo "Base stable tag: ${BASE_TAG_STABLE}"

# ---- build derived (ROS2 desktop + Orbbec ROS2 wrapper) ----
if [ ! -f "${DERIVED_DOCKERFILE}" ]; then
  echo "ERROR: ${DERIVED_DOCKERFILE} not found."
  echo "Tip: If you only want base, stop here."
  exit 1
fi

echo "[2/2] Build derived image (ROS2 desktop + Orbbec): ${DERIVED_TAG_DATE}"
docker build \
  -f "${DERIVED_DOCKERFILE}" \
  --build-arg BASE_IMAGE="${BASE_TAG_STABLE}" \
  -t "${DERIVED_TAG_DATE}" \
  --label ade_image_commit_sha="${GIT_SHA}" \
  --label ade_image_commit_tag="${TAG_TIME}" \
  "${CONTEXT}"

# stable tag for ADE usage
docker tag "${DERIVED_TAG_DATE}" "${DERIVED_TAG_STABLE}"
echo "Derived stable tag: ${DERIVED_TAG_STABLE}"

# ---- cleanup dangling images ----
dangling_images=$(docker images -f "dangling=true" -q)
if [ -n "$dangling_images" ]; then
  docker rmi -f $dangling_images || true
fi

echo ""
echo "Docker images built successfully:"
echo "  - ${BASE_TAG_DATE}"
echo "  - ${BASE_TAG_STABLE}"
echo "  - ${DERIVED_TAG_DATE}"
echo "  - ${DERIVED_TAG_STABLE}"
