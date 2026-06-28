#!/usr/bin/env python3
"""从 replace_all2.log 回填 replace_progress.txt 的 done-set。
复用 3_1_scene_split._is_terminal_status / _stem_of (DRY): 只把终态 stem 写入,
可恢复失败 (拉取失败/PUSH-FAIL/...) 不写 -> 续跑自动重试。去重保序。

用法:
  python3 tools/backfill_replace_progress.py [LOG] [OUT]
默认: LOG=data/logs/replace_all2.log  OUT=data/pipeline_state/3_replace_progress.txt (相对 videos/ 目录)。
"""
import os
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEOS = os.path.dirname(HERE)


def _load_split_mod():
    p = os.path.join(VIDEOS, "3_1_scene_split.py")
    spec = importlib.util.spec_from_file_location("scene_split", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def backfill(log_path: str, out_path: str) -> int:
    """读 log, 取终态 stem 去重写 out。返回写入条数。"""
    m = _load_split_mod()
    seen, ordered = set(), []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("═══"):
                continue
            if m._is_terminal_status(line):
                stem = m._stem_of(line)
                if stem not in seen:
                    seen.add(stem); ordered.append(stem)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(s + "\n" for s in ordered))
    return len(ordered)


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIDEOS, "data", "logs", "replace_all2.log")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(VIDEOS, "data", "pipeline_state", "3_replace_progress.txt")
    n = backfill(log, out)
    print(f"backfilled {n} terminal stems -> {out}")


if __name__ == "__main__":
    main()
