"""
一键运行脚本 - 百万级视频 ID 采集

使用方式:
  # 在能访问 YouTube 的机器上运行全量采集:
  python3 run_all.py --all

  # 仅运行数据集获取 (不需要 YouTube 访问):
  python3 run_all.py --datasets

  # 仅运行频道爬取 (需要 YouTube 访问):
  python3 run_all.py --channels

  # 仅运行关键词搜索 (需要 YouTube 访问):
  python3 run_all.py --search

  # 仅合并结果:
  python3 run_all.py --merge

代理设置:
  export YT_PROXY=http://your-proxy:port/   # YouTube 访问代理
  export GITHUB_PROXY=http://your-proxy:port/  # GitHub/S3 数据集下载代理
"""
import sys
import subprocess
import argparse
from pathlib import Path

PYTHON = sys.executable
BASE = Path(__file__).parent


def run_script(name, desc):
    script = BASE / name
    print(f"\n{'='*60}")
    print(f"  [{desc}] 运行 {name}")
    print(f"{'='*60}\n")
    result = subprocess.run([PYTHON, str(script)], cwd=str(BASE))
    if result.returncode != 0:
        print(f"  [警告] {name} 退出码: {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="百万级视频 ID 采集")
    parser.add_argument("--all", action="store_true", help="运行全部流程")
    parser.add_argument("--datasets", action="store_true", help="仅获取公开数据集")
    parser.add_argument("--channels", action="store_true", help="仅频道爬取")
    parser.add_argument("--search", action="store_true", help="仅关键词搜索")
    parser.add_argument("--merge", action="store_true", help="仅合并结果")
    parser.add_argument("--discover", action="store_true", help="仅频道发现")
    args = parser.parse_args()

    # 默认 --all
    if not any([args.all, args.datasets, args.channels, args.search, args.merge, args.discover]):
        args.all = True

    if args.all or args.discover:
        run_script("discover_channels.py", "频道发现")

    if args.all or args.datasets:
        run_script("fetch_datasets.py", "公开数据集获取")

    if args.all or args.search:
        run_script("search_videos.py", "关键词搜索")

    if args.all or args.channels:
        run_script("crawl_channels.py", "频道爬取")

    if args.all or args.merge:
        run_script("merge_results.py", "合并去重")

    print(f"\n{'='*60}")
    print("  全部完成! 查看结果: results/all_video_ids.jsonl")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
