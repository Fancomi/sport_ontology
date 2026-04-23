#!/bin/bash
# 关闭所有占用 GPU 的进程（通过 fuser /dev/nvidia*）

pids=$(fuser -v /dev/nvidia* 2>&1 | awk '/^[[:space:]]/ && $2 ~ /^[0-9]+$/ {print $2}' | sort -u)

if [ -z "$pids" ]; then
    echo "未发现占用 GPU 的进程"
    exit 0
fi

count=$(echo "$pids" | wc -w)
echo "发现 $count 个占用 GPU 的进程，正在关闭: $pids"
kill $pids
sleep 2

remaining=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u)
if [ -z "$remaining" ]; then
    echo "全部关闭完成"
else
    echo "强制 kill -9: $remaining"
    kill -9 $remaining
    echo "完成"
fi
