#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
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

if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
else
  GIT_SHA="nogit"
fi

BASE_DOCKERFILE="${REPO_ROOT}/${ARCH}.base.dockerfile"
DERIVED_DOCKERFILE="${REPO_ROOT}/${ARCH}.orbbec_ros2_driver.dockerfile"
L3_DOCKERFILE="${REPO_ROOT}/${ARCH}_l3_envsdk_requirements.dockerfile"

# Only stable tags (no date tags)
BASE_TAG="openorbbecsdk-env:${ARCH}_base"
DERIVED_TAG="openorbbecsdk-env:${ARCH}_ros2_orbbec_latest"
L3_TAG="openorbbecsdk-env:${ARCH}_ros2_orbbec_sdk_latest"

# ---- build base ----
if [ ! -f "${BASE_DOCKERFILE}" ]; then
  echo "ERROR: ${BASE_DOCKERFILE} not found."
  exit 1
fi

echo "[1/3] Build base image: ${BASE_TAG}"
docker build \
  -f "${BASE_DOCKERFILE}" \
  -t "${BASE_TAG}" \
  --label ade_image_commit_sha="${GIT_SHA}" \
  "${CONTEXT}"

# ---- build derived ----
if [ ! -f "${DERIVED_DOCKERFILE}" ]; then
  echo "ERROR: ${DERIVED_DOCKERFILE} not found."
  exit 1
fi

echo "[2/3] Build derived image (ROS2+Orbbec): ${DERIVED_TAG}"
docker build \
  -f "${DERIVED_DOCKERFILE}" \
  --build-arg BASE_IMAGE="${BASE_TAG}" \
  -t "${DERIVED_TAG}" \
  --label ade_image_commit_sha="${GIT_SHA}" \
  "${CONTEXT}"

# ---- build L3 ----
if [ -f "${L3_DOCKERFILE}" ]; then
  echo "[3/3] Build L3 image (Algo/SDK deps): ${L3_TAG}"
  docker build \
    -f "${L3_DOCKERFILE}" \
    -t "${L3_TAG}" \
    --label ade_image_commit_sha="${GIT_SHA}" \
    "${CONTEXT}"
else
  echo "[3/3] Skip L3: ${L3_DOCKERFILE} not found."
fi

# ---- optional cleanup: remove intermediate build cache layers ----
# docker builder prune -f

echo ""
echo "Docker images built successfully:"
echo "  - ${BASE_TAG}"
echo "  - ${DERIVED_TAG}"
if [ -f "${L3_DOCKERFILE}" ]; then
  echo "  - ${L3_TAG}"
fi
