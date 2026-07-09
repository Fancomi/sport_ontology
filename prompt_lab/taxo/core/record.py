"""图记录 append-only JSONL + 续跑游标。中断可续, 不改写历史行。"""
import json
from pathlib import Path


def append(path: Path, row: dict) -> None:
    """追加一条记录(一行一 JSON)。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_all(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text("utf-8").splitlines() if line.strip()]


def done_ids(path: Path) -> set[str]:
    """已写过的 image_id 集合(续跑去重用)。"""
    return {r["image_id"] for r in read_all(path)}


def save_cursor(path: Path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")


def load_cursor(path: Path, default: dict | None = None) -> dict:
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    return json.loads(p.read_text("utf-8"))
