"""归一化: raw_value -> canonical_value。纯函数, 可回放。

两步: (1) normalize 做通用清洗(小写/去首尾标点/压空白);
      (2) apply_map 查该 key 的同义映射表(canon_map)。
canon_map 结构: {key_id: {raw_or_normalized: canonical}}, 单独版本化落盘。
"""
import json
import re
from pathlib import Path

_WS = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[\s.,;:!?'\"()\[\]]+|[\s.,;:!?'\"()\[\]]+$")


def normalize(value: str) -> str:
    """通用清洗: 转小写、去首尾标点/空白、内部空白压成单空格。"""
    if value is None:
        return ""
    v = str(value).lower()
    v = _EDGE_PUNCT.sub("", v)
    v = _WS.sub(" ", v).strip()
    return v


def apply_map(key_id: str, value: str, cmap: dict) -> str:
    """先 normalize, 再查 cmap[key_id]。映射表的键也按 normalize 后匹配。"""
    norm = normalize(value)
    key_map = cmap.get(key_id, {})
    # 映射表键统一 normalize 后比对, 保证 "小狗"/"Small Dog" 都能命中
    for raw, canonical in key_map.items():
        if normalize(raw) == norm:
            return canonical
    return norm


def canonicalize_json(json_raw: dict, cmap: dict) -> dict:
    """对整条 JSON 做归一化, 返回 json_canon。"""
    return {k: apply_map(k, v, cmap) for k, v in json_raw.items()}


def load_map(path: Path) -> dict:
    """读 canon_map.vN.json; 不存在则返回空表。"""
    p = Path(path)
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def save_map(cmap: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(cmap, ensure_ascii=False, indent=2), "utf-8")
