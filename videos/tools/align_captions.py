#!/usr/bin/env python3
"""caption 三方对齐到权威名单 canonical_segments.list。

口径: 磁盘 caption JSON 与 canonical 严格对齐。
  - 孤儿 (磁盘有∖canonical无): mv 到 captions/_orphan/<shard>/ (可逆, 不真删)
  - 缺口 (canonical有∖磁盘无): 写 4_to_caption.list 待办 (不重跑 caption)
  - 重建标记 4_caption_progress.txt = 对齐后磁盘真相 (= 交集)

用法:
  python3 tools/align_captions.py            # dry-run, 只报告
  python3 tools/align_captions.py --apply     # 执行: 移孤儿 + 重建标记 + 写缺口
"""
import os, sys, argparse, hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import config

CANONICAL = config.DELIVERABLES_DIR / "3_canonical_segments.list"
PROGRESS  = config.STATE_DIR / "4_caption_progress.txt"
TO_CAPTION = config.STATE_DIR / "4_to_caption.list"
ORPHAN_MOVED = config.STATE_DIR / "4_captions_orphan_moved.list"
CAP_DIR = config.DATA_DIR / "captions"


def _strip_mp4(name: str) -> str:
    """canonical 名单条目带 .mp4 后缀, caption JSON 的 stem 不带 —— 统一去后缀比对。"""
    return name[:-4] if name.endswith(".mp4") else name


def plan_alignment(canonical: set, disk_stems: set) -> dict:
    """纯函数: 给定 canonical 与磁盘 stem 集, 算出对齐计划。
    canonical 可能带 .mp4 后缀 (切片名单), disk_stems 不带 (JSON 文件名去 .json);
    两侧都规范化去 .mp4 后按 stem 比对。返回的集合统一为无 .mp4 的 stem。"""
    canon = {_strip_mp4(s) for s in canonical}
    disk  = {_strip_mp4(s) for s in disk_stems}
    aligned = canon & disk          # 两边都有 -> 保留, 即最终 caption 集
    orphans = disk - canon          # 磁盘有但不在权威 -> 移走
    gap     = canon - disk          # 权威有但磁盘无 -> 待 caption
    return {"aligned": aligned, "orphans": orphans, "gap": gap}


def scan_disk_stems(cap_dir: str) -> set:
    """扫 captions/<shard>/*.json 取 stem (文件名去 .json)。跳过 _orphan/ 子树。"""
    stems = set()
    root = Path(cap_dir)
    if not root.exists():
        return stems
    for shard in root.iterdir():
        if not shard.is_dir() or shard.name == "_orphan":
            continue
        for j in shard.glob("*.json"):
            stems.add(j.stem)
    return stems


def move_orphans(cap_dir: str, orphans: set) -> int:
    """把孤儿 JSON mv 到 cap_dir/_orphan/<shard>/ (保留分片路径)。返回移动数。"""
    root = Path(cap_dir)
    moved = 0
    for stem in orphans:
        shard = hashlib.md5(stem.encode()).hexdigest()[:2]
        src = root / shard / f"{stem}.json"
        if not src.exists():
            continue
        dst_dir = root / "_orphan" / shard
        dst_dir.mkdir(parents=True, exist_ok=True)
        src.rename(dst_dir / f"{stem}.json")
        moved += 1
    return moved


def _read_set(p: Path) -> set:
    return {l.strip() for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()


def _write_sorted(p: Path, s: set):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("\n".join(sorted(s)) + ("\n" if s else ""))
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="执行移动/重建 (默认 dry-run)")
    args = ap.parse_args()

    if not CANONICAL.exists():
        sys.exit(f"缺权威名单: {CANONICAL}")
    canonical = _read_set(CANONICAL)               # 带 .mp4
    print(f"扫描磁盘 caption JSON ({CAP_DIR}) ...", flush=True)
    disk = scan_disk_stems(str(CAP_DIR))           # 不带 .mp4
    old_prog = _read_set(PROGRESS)
    plan = plan_alignment(canonical, disk)          # 返回规范化(无 .mp4)的 stem 集
    canon_norm = {_strip_mp4(s) for s in canonical}
    disk_norm  = {_strip_mp4(s) for s in disk}

    print(f"\n═══ caption 对账报告 ═══")
    print(f"权威 canonical:        {len(canonical):>9}")
    print(f"磁盘实际 JSON:         {len(disk):>9}")
    print(f"旧标记 caption_progress:{len(old_prog):>9}  (过时, 将被磁盘真相覆盖)")
    print(f"── 对齐后保留 (交集):  {len(plan['aligned']):>9} ──")
    print(f"孤儿 (磁盘∖权威, 待移): {len(plan['orphans']):>9}")
    print(f"缺口 (权威∖磁盘, 待审): {len(plan['gap']):>9}")
    assert plan["aligned"] | plan["gap"] == canon_norm, "校验失败: aligned+gap != canonical"
    assert plan["aligned"] == disk_norm - plan["orphans"], "校验失败: aligned != disk-orphans"
    print("校验: aligned+gap==canonical OK, aligned==disk-orphans OK")

    if not args.apply:
        print("\n[dry-run] 未改动。加 --apply 执行移孤儿 + 重建标记 + 写缺口。")
        return

    print(f"\n移动 {len(plan['orphans'])} 孤儿 -> {CAP_DIR}/_orphan/ ...", flush=True)
    moved = move_orphans(str(CAP_DIR), plan["orphans"])
    # 标记/缺口/孤儿清单统一写回带 .mp4 的切片名 (与 4_caption.py 读 canonical 比对的形式一致)
    _write_sorted(ORPHAN_MOVED, {s + ".mp4" for s in plan["orphans"]})
    _write_sorted(PROGRESS, {s + ".mp4" for s in plan["aligned"]})
    _write_sorted(TO_CAPTION, {s + ".mp4" for s in plan["gap"]})
    print(f"已移孤儿: {moved}")
    print(f"重建标记: {PROGRESS} = {len(plan['aligned'])}")
    print(f"缺口待办: {TO_CAPTION} = {len(plan['gap'])}")
    print(f"孤儿清单: {ORPHAN_MOVED} (抽查 _orphan/ 确认后可单独真删)")


if __name__ == "__main__":
    main()
