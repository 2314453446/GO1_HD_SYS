#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export ADE_NAME=camera

# 1) 已运行：直接进入
if ade status >/dev/null 2>&1; then
  exec ade enter
fi
if docker ps --format '{{.Names}}' | grep -qx "camera"; then
  exec ade enter
fi

# 2) 已存在但退出：启动“同一个”容器，再 ade 进入（复用关键）
if docker ps -a --format '{{.Names}}' | grep -qx "camera"; then
  docker start camera >/dev/null
  exec ade enter
fi

# 3) 容器不存在：第一次创建才 start（不要默认 --update）
ade start
exec ade enter


