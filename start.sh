#!/bin/bash
# 启动脚本（调用system中的实际脚本）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
./system/start.sh "$@"
