#!/bin/bash
# 百万级视频 ID 采集 - 一键并行启动 (多样性优先)
# 用法: bash run.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/python3"
LOGS="$SCRIPT_DIR/logs"
mkdir -p "$LOGS" results datasets

# 代理配置
export http_proxy=http://agent.baidu.com:8188
export https_proxy=http://agent.baidu.com:8188
export YT_PROXY=http://agent.baidu.com:8188
export GITHUB_PROXY=http://njxg-banqian20230721-sousuo00230.njxg:3231/

echo "=========================================="
echo " 视频 ID 采集 - 多样性优先 - 并行启动"
echo "=========================================="
echo " 代理: $YT_PROXY"
echo " Python: $PYTHON"
echo ""

# Step 1: 频道发现 (快速同步)
echo "[1/5] 频道发现..."
$PYTHON discover_channels.py 2>&1 | tail -3
echo ""

# Step 2: 并行启动四条管线
echo "[2/5] 并行启动采集管线..."

# 管线A: 数据集全量获取 (Kinetics 65万)
echo "  → A. 数据集获取"
$PYTHON fetch_datasets.py > "$LOGS/datasets_run.log" 2>&1 &
PID_A=$!

# 管线B: 频道爬取 (每频道限50条)
echo "  → B. 频道爬取"
$PYTHON crawl_channels.py > "$LOGS/crawl_run.log" 2>&1 &
PID_B=$!

# 管线C: 关键词搜索
echo "  → C. 关键词搜索"
$PYTHON search_videos.py > "$LOGS/search_run.log" 2>&1 &
PID_C=$!

# 管线D: 多样性搜索 (多语言 × 参数轮换 × 频道配额)
echo "  → D. 多样性搜索 (核心)"
$PYTHON diverse_crawl.py > "$LOGS/diverse_run.log" 2>&1 &
PID_D=$!

echo ""
echo "  PID: A=$PID_A B=$PID_B C=$PID_C D=$PID_D"
echo ""

# Step 3: 等待
echo "[3/5] 等待所有管线完成..."
echo "  监控: tail -f logs/diverse_crawl.log"
echo ""

wait $PID_A 2>/dev/null && echo "  ✓ A. 数据集完成" || echo "  ✗ A. 数据集异常"
wait $PID_C 2>/dev/null && echo "  ✓ C. 搜索完成" || echo "  ✗ C. 搜索异常"
wait $PID_B 2>/dev/null && echo "  ✓ B. 频道爬取完成" || echo "  ✗ B. 频道爬取异常"
wait $PID_D 2>/dev/null && echo "  ✓ D. 多样性搜索完成" || echo "  ✗ D. 多样性搜索异常"
echo ""

# Step 4: 合并去重
echo "[4/5] 合并去重..."
rm -f results/all_video_ids.jsonl
$PYTHON merge_results.py 2>&1 | tail -12
echo ""

# Step 5: 统计
echo "[5/5] 多样性统计..."
$PYTHON -c "
import json
from collections import Counter
items = []
with open('results/all_video_ids.jsonl') as f:
    for line in f:
        items.append(json.loads(line))
channels = Counter(r.get('channel','unknown') for r in items)
sources = Counter(r.get('source','unknown') for r in items)
print(f'总视频 ID: {len(items)}')
print(f'不重复频道: {len(channels)}')
print(f'来源分布:')
for s, c in sources.most_common():
    print(f'  {s}: {c}')
print(f'频道分布: top1={channels.most_common(1)[0][1]} 条, 中位数频道贡献={sorted(channels.values())[len(channels)//2]} 条')
"
echo ""
echo "=========================================="
echo " 完成! 结果: results/all_video_ids.jsonl"
echo "=========================================="
wc -l results/all_video_ids.jsonl
