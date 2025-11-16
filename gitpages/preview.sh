#!/bin/bash
# 本地预览脚本

echo "正在启动本地预览服务器..."
PORT=3002
echo "访问地址: http://localhost:${PORT}"
echo "按 Ctrl+C 停止服务器"
echo ""

# 检查是否安装了Python
if command -v python3 &> /dev/null; then
    echo "使用 Python 3 启动服务器..."
    cd "$(dirname "$0")/.."
    python3 -m http.server "${PORT}"
elif command -v python &> /dev/null; then
    echo "使用 Python 启动服务器..."
    cd "$(dirname "$0")/.."
    python -m http.server "${PORT}"
else
    echo "错误: 未找到 Python，请先安装 Python"
    echo ""
    echo "或者使用 Node.js 的 http-server:"
    echo "  npm install -g http-server"
    echo "  http-server -p 3000"
    exit 1
fi

