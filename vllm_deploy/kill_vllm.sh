#!/bin/bash
# 一键关闭所有 vllm serve 进程

pids=$(pgrep -f "vllm serve" 2>/dev/null)
if [ -z "$pids" ]; then
    echo "未发现运行中的 vllm serve 进程"
    exit 0
fi

count=$(echo "$pids" | wc -w)
echo "发现 $count 个 vllm serve 进程，正在关闭..."
kill $pids
sleep 2

# 确认是否全部退出
remaining=$(pgrep -f "vllm serve" 2>/dev/null)
if [ -z "$remaining" ]; then
    echo "全部关闭完成"
else
    echo "强制 kill -9..."
    kill -9 $remaining
    echo "完成"
fi
