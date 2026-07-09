# 迭代式 Taxonomy 发现闭环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个多轮迭代闭环,用 VLM 从 COCO 子集抽 caption+JSON→检测 label_set 碰撞→Opus 裂簇提新 Key→ontology 沉淀→双闸终止,产出一套能区分图像的 taxonomy(Schema)。

**Architecture:** 方案 A(扁平复选标签+碰撞驱动细化)。纯函数核心(schema/canon/collide/record)+ 可插拔后端(source/extractor/judge/reviewer)+ 编排层(run_round/loop)。全部落在 `prompt_lab/taxo/`,与主工程解耦。gemma-8001 当抽取学生,Opus 4.8 当 teacher/judge。

**Tech Stack:** Python 3.11(dino venv), dspy 3.2.1, PIL, pytest, 本地 vLLM(gemma vision, OpenAI 兼容), Opus 4.8(Anthropic 兼容, 走 urllib 裸 HTTP,无需 anthropic SDK)。

**规范约定(全局)**
- 环境: `source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate`
- 工作目录: `cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/prompt_lab`
- 所有相对路径以 `prompt_lab/` 为根。
- 测试运行: `python -m pytest taxo/tests/<file>::<name> -v`
- 每个 Key 用稳定 ID `k_NNN`(三位零填充,从 `k_000` 起)。

---

## 文件结构

**核心纯函数(无外部依赖,可单测):**
- `taxo/core/schema.py` — Schema Registry: Key 增/合并/软删 + 版本快照 vN.json/HEAD
- `taxo/core/canon.py` — 归一化纯函数 + canon_map 版本化
- `taxo/core/collide.py` — label_set→指纹→碰撞分桶
- `taxo/core/record.py` — 图记录 append-only JSONL 读写 + 续跑游标

**可插拔后端(各封装一种外部依赖):**
- `taxo/backends/source.py` — ImageSource: COCO loader → (image_id, image_bytes, gt)
- `taxo/backends/extractor.py` — Extractor: dspy 驱动 gemma vision 抽 caption+JSON
- `taxo/backends/judge.py` — Judge: Opus 4.8 裂簇/合并/打分(裸 HTTP + 缓存)
- `taxo/backends/reviewer.py` — HTML 产出 + 可选人工门

**编排:**
- `taxo/config.py` — 数据源/端点/轮次/scope/阈值/权重,一处配置
- `taxo/metrics.py` — 四分量无监督 metric + 每轮指标聚合
- `taxo/run_round.py` — 单轮编排
- `taxo/loop.py` — 多轮驱动 + 双闸终止 + 续跑

**测试:** `taxo/tests/test_*.py`(镜像 `tools/tests/` 风格)

---

## Task 0: 项目骨架与依赖

**Files:**
- Create: `taxo/__init__.py`, `taxo/core/__init__.py`, `taxo/backends/__init__.py`, `taxo/tests/__init__.py`
- Create: `taxo/config.py`
- Modify: `.gitignore`(追加 `prompt_lab/taxo/runs/`)

- [ ] **Step 1: 安装 pytest(内网源)**

Run:
```bash
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
pip install pytest -i https://pip.baidu-int.com/simple/
```
Expected: 成功安装 pytest(dspy/PIL 已在)。

- [ ] **Step 2: 建包目录**

Run:
```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/prompt_lab
mkdir -p taxo/core taxo/backends taxo/tests
touch taxo/__init__.py taxo/core/__init__.py taxo/backends/__init__.py taxo/tests/__init__.py
```

- [ ] **Step 3: 写 config.py**

Create `taxo/config.py`:
```python
"""taxo 全局配置。一处改, 全局生效。换数据源/端点/阈值都在这里。"""
from pathlib import Path

TAXO_DIR = Path(__file__).resolve().parent
RUNS_DIR = TAXO_DIR / "runs"

# ── 数据源 ────────────────────────────────────────────────
DATA_ROOT = Path("/root/paddlejob/workspace/env_run/penghaotian/datas")
COCO_SPLIT = "val2014"          # val 集较小, 原型够用
COCO_IMAGES = DATA_ROOT / "coco" / "images" / COCO_SPLIT
COCO_CAPTIONS = DATA_ROOT / "coco" / "annotations" / "captions_val2014.json"
COCO_INSTANCES = DATA_ROOT / "coco" / "annotations" / "instances_val2014.json"
SUBSET_SIZE = 500               # 原型子集大小
SUBSET_SEED = 42

# ── 端点 ─────────────────────────────────────────────────
VLM_MODEL = "/dev/shm/models/gemma-4-26B-A4B-it"
VLM_BASE = "http://127.0.0.1:8001/v1"
CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
JUDGE_MODEL = "Opus 4.8"

# ── 闭环参数 ──────────────────────────────────────────────
COLLISION_SCOPE = "incremental"   # incremental | global
MAX_ROUNDS = 6
MAX_KEYS = 40                     # Schema Key 数上限(安全阀)
REOPT_KEY_THRESHOLD = 3           # 新增 Key ≥ 此值才重跑 GEPA
STABILITY_REPEATS = 2             # 同图重抽次数(算 stability)

# 双闸: 收敛判据
CONVERGE_WINDOW = 2               # 连续 N 轮
CONVERGE_MAX_NEW_KEYS = 1         # 新增 Key ≤ 此值
CONVERGE_MIN_DROP_RATE = 0.10     # 碰撞簇下降率 < 此值

# metric 四分量权重(等权起步)
METRIC_WEIGHTS = {"stability": 0.25, "validity": 0.25,
                  "coverage": 0.25, "faithfulness": 0.25}

# review
REVIEW_MODE = "off"               # off | on
REVIEW_TIMEOUT_S = 0              # on 模式下等待 review.json 的超时(0=不超时)
```

- [ ] **Step 4: 更新 .gitignore**

在 `.gitignore` 追加一行(找到 videos 运行态那段附近):
```
prompt_lab/taxo/runs/
```

- [ ] **Step 5: 冒烟验证导入**

Run:
```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/prompt_lab
python -c "from taxo import config; print(config.SUBSET_SIZE, config.JUDGE_MODEL)"
```
Expected: `500 Opus 4.8`

- [ ] **Step 6: Commit**

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology
git add prompt_lab/taxo .gitignore
git commit -m "feat(taxo): 项目骨架 + config"
```

---

## Task 1: core/collide.py — label_set 指纹与碰撞分桶

**Files:**
- Create: `taxo/core/collide.py`
- Test: `taxo/tests/test_collide.py`

- [ ] **Step 1: 写失败测试**

Create `taxo/tests/test_collide.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from taxo.core import collide


def test_fingerprint_is_order_independent():
    a = collide.fingerprint({"k_001": "dog", "k_002": "outdoor"})
    b = collide.fingerprint({"k_002": "outdoor", "k_001": "dog"})
    assert a == b


def test_fingerprint_skips_empty_values():
    a = collide.fingerprint({"k_001": "dog", "k_002": ""})
    b = collide.fingerprint({"k_001": "dog"})
    assert a == b


def test_find_collisions_groups_identical():
    records = [
        {"image_id": "a", "label_set_fp": collide.fingerprint({"k_001": "dog"})},
        {"image_id": "b", "label_set_fp": collide.fingerprint({"k_001": "dog"})},
        {"image_id": "c", "label_set_fp": collide.fingerprint({"k_001": "cat"})},
    ]
    clusters = collide.find_collisions(records)
    assert len(clusters) == 1
    assert set(clusters[0]["image_ids"]) == {"a", "b"}


def test_find_collisions_ignores_singletons():
    records = [
        {"image_id": "a", "label_set_fp": collide.fingerprint({"k_001": "dog"})},
        {"image_id": "b", "label_set_fp": collide.fingerprint({"k_001": "cat"})},
    ]
    assert collide.find_collisions(records) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_collide.py -v`
Expected: FAIL(`ModuleNotFoundError` 或 `AttributeError: fingerprint`)

- [ ] **Step 3: 实现 collide.py**

Create `taxo/core/collide.py`:
```python
"""label_set 指纹 + 碰撞分桶。纯函数, 无外部依赖。

指纹 = 对非空 (key_id, value) 排序后 hash → 相同指纹即碰撞。
"""
import hashlib


def label_pairs(json_canon: dict) -> list[tuple[str, str]]:
    """取非空值的 (key_id, value), 按 key_id 排序。空值(''/None)剔除。"""
    pairs = [(k, str(v)) for k, v in json_canon.items() if v not in ("", None)]
    return sorted(pairs)


def fingerprint(json_canon: dict) -> str:
    """稳定指纹: 排序后 (key,value) 拼接的 sha1。顺序无关, 忽略空值。"""
    pairs = label_pairs(json_canon)
    raw = "|".join(f"{k}={v}" for k, v in pairs)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def find_collisions(records: list[dict]) -> list[dict]:
    """records 需含 image_id + label_set_fp。返回 size>=2 的簇。

    返回: [{"fp": ..., "image_ids": [...]}], 按簇大小降序。
    """
    buckets: dict[str, list[str]] = {}
    for r in records:
        buckets.setdefault(r["label_set_fp"], []).append(r["image_id"])
    clusters = [{"fp": fp, "image_ids": ids}
                for fp, ids in buckets.items() if len(ids) >= 2]
    clusters.sort(key=lambda c: len(c["image_ids"]), reverse=True)
    return clusters
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_collide.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add prompt_lab/taxo/core/collide.py prompt_lab/taxo/tests/test_collide.py
git commit -m "feat(taxo): collide 指纹与碰撞分桶 + 测试"
```

---

## Task 2: core/canon.py — 归一化纯函数 + 版本化映射表

**Files:**
- Create: `taxo/core/canon.py`
- Test: `taxo/tests/test_canon.py`

- [ ] **Step 1: 写失败测试**

Create `taxo/tests/test_canon.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from taxo.core import canon


def test_normalize_lowercases_and_strips():
    assert canon.normalize("  Dog. ") == "dog"


def test_normalize_collapses_whitespace():
    assert canon.normalize("small   dog") == "small dog"


def test_apply_map_maps_synonym():
    cmap = {"k_001": {"小狗": "dog", "small dog": "dog"}}
    assert canon.apply_map("k_001", "小狗", cmap) == "dog"


def test_apply_map_passthrough_when_no_entry():
    cmap = {"k_001": {"小狗": "dog"}}
    assert canon.apply_map("k_001", "Cat", cmap) == "cat"


def test_canonicalize_json_full():
    cmap = {"k_001": {"小狗": "dog"}}
    out = canon.canonicalize_json({"k_001": "小狗", "k_002": " Outdoor "}, cmap)
    assert out == {"k_001": "dog", "k_002": "outdoor"}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_canon.py -v`
Expected: FAIL(`AttributeError: normalize`)

- [ ] **Step 3: 实现 canon.py**

Create `taxo/core/canon.py`:
```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_canon.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add prompt_lab/taxo/core/canon.py prompt_lab/taxo/tests/test_canon.py
git commit -m "feat(taxo): canon 归一化 + 版本化映射表 + 测试"
```

---

## Task 3: core/schema.py — Schema Registry(版本化, 软删)

**Files:**
- Create: `taxo/core/schema.py`
- Test: `taxo/tests/test_schema.py`

- [ ] **Step 1: 写失败测试**

Create `taxo/tests/test_schema.py`:
```python
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pathlib import Path
from taxo.core import schema


def _reg(tmp):
    return schema.SchemaRegistry(Path(tmp))


def test_add_key_assigns_sequential_ids():
    with tempfile.TemporaryDirectory() as tmp:
        r = _reg(tmp)
        k1 = r.add_key(name="primary_object", desc="主体", value_type="open",
                       introduced_round=0, introduced_by="seed")
        k2 = r.add_key(name="scene", desc="场景", value_type="enum",
                       allowed_values=["indoor", "outdoor"],
                       introduced_round=0, introduced_by="seed")
        assert k1 == "k_000"
        assert k2 == "k_001"


def test_active_keys_excludes_soft_deleted():
    with tempfile.TemporaryDirectory() as tmp:
        r = _reg(tmp)
        a = r.add_key(name="a", desc="", value_type="open",
                      introduced_round=0, introduced_by="seed")
        b = r.add_key(name="b", desc="", value_type="open",
                      introduced_round=0, introduced_by="seed")
        r.merge_key(b, into=a)
        active = [k["id"] for k in r.active_keys()]
        assert a in active and b not in active


def test_snapshot_and_reload_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        r = _reg(tmp)
        r.add_key(name="a", desc="", value_type="open",
                  introduced_round=0, introduced_by="seed")
        v = r.snapshot()                    # 返回版本号
        r2 = schema.SchemaRegistry(Path(tmp))   # 从 HEAD 重新加载
        assert [k["name"] for k in r2.active_keys()] == ["a"]
        assert r2.version == v


def test_keys_over_limit_detection():
    with tempfile.TemporaryDirectory() as tmp:
        r = _reg(tmp)
        for i in range(3):
            r.add_key(name=f"k{i}", desc="", value_type="open",
                      introduced_round=0, introduced_by="seed")
        assert r.n_active() == 3
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_schema.py -v`
Expected: FAIL(`AttributeError: SchemaRegistry`)

- [ ] **Step 3: 实现 schema.py**

Create `taxo/core/schema.py`:
```python
"""Schema Registry: 版本化的 Key 集合。Key 只增不物理删(合并=软删)。

落盘: <dir>/schema/vN.json (整份快照) + <dir>/schema/HEAD (当前版本号)。
主键是稳定 ID k_NNN, 不随 name 变。
"""
import json
from pathlib import Path


class SchemaRegistry:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "schema"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keys: dict[str, dict] = {}
        self.version = -1
        self._load_head()

    # ── 持久化 ────────────────────────────────────────────
    def _head_file(self) -> Path:
        return self.dir / "HEAD"

    def _load_head(self) -> None:
        head = self._head_file()
        if head.exists():
            self.version = int(head.read_text().strip())
            data = json.loads((self.dir / f"v{self.version}.json").read_text("utf-8"))
            self.keys = {k["id"]: k for k in data["keys"]}

    def snapshot(self) -> int:
        """写下一版快照, 更新 HEAD, 返回新版本号。"""
        self.version += 1
        data = {"version": self.version, "keys": list(self.keys.values())}
        (self.dir / f"v{self.version}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        self._head_file().write_text(str(self.version))
        return self.version

    # ── Key 操作 ──────────────────────────────────────────
    def _next_id(self) -> str:
        return f"k_{len(self.keys):03d}"

    def add_key(self, *, name: str, desc: str, value_type: str,
                introduced_round: int, introduced_by: str,
                allowed_values: list | None = None,
                parent: str | None = None) -> str:
        kid = self._next_id()
        self.keys[kid] = {
            "id": kid, "name": name, "desc": desc, "value_type": value_type,
            "allowed_values": allowed_values or [], "parent": parent,
            "synonyms_of": None,
            "introduced_round": introduced_round, "introduced_by": introduced_by,
        }
        return kid

    def merge_key(self, kid: str, *, into: str) -> None:
        """软删 kid, 指向 into。历史引用不受影响。"""
        self.keys[kid]["synonyms_of"] = into

    def active_keys(self) -> list[dict]:
        return [k for k in self.keys.values() if k["synonyms_of"] is None]

    def n_active(self) -> int:
        return len(self.active_keys())

    def get(self, kid: str) -> dict | None:
        return self.keys.get(kid)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_schema.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add prompt_lab/taxo/core/schema.py prompt_lab/taxo/tests/test_schema.py
git commit -m "feat(taxo): Schema Registry 版本化+软删 + 测试"
```

---

## Task 4: core/record.py — 图记录 append-only + 续跑游标

**Files:**
- Create: `taxo/core/record.py`
- Test: `taxo/tests/test_record.py`

- [ ] **Step 1: 写失败测试**

Create `taxo/tests/test_record.py`:
```python
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pathlib import Path
from taxo.core import record


def test_append_and_read_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "records.jsonl"
        record.append(f, {"image_id": "a", "round": 0})
        record.append(f, {"image_id": "b", "round": 0})
        rows = record.read_all(f)
        assert [r["image_id"] for r in rows] == ["a", "b"]


def test_done_ids_dedups_across_append():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "records.jsonl"
        record.append(f, {"image_id": "a", "round": 0})
        record.append(f, {"image_id": "a", "round": 0})
        assert record.done_ids(f) == {"a"}


def test_cursor_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "state.json"
        record.save_cursor(f, {"last_round": 2, "pending": ["x"]})
        assert record.load_cursor(f) == {"last_round": 2, "pending": ["x"]}


def test_load_cursor_default_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "state.json"
        assert record.load_cursor(f, default={"last_round": -1}) == {"last_round": -1}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_record.py -v`
Expected: FAIL(`AttributeError: append`)

- [ ] **Step 3: 实现 record.py**

Create `taxo/core/record.py`:
```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_record.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add prompt_lab/taxo/core/record.py prompt_lab/taxo/tests/test_record.py
git commit -m "feat(taxo): 图记录 append-only + 续跑游标 + 测试"
```

---

## Task 5: backends/source.py — COCO ImageSource

**Files:**
- Create: `taxo/backends/source.py`
- Test: `taxo/tests/test_source.py`

- [ ] **Step 1: 写失败测试**

Create `taxo/tests/test_source.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import pytest
from taxo.backends import source
from taxo import config


def test_coco_source_subset_size_and_seed_stable():
    s = source.CocoSource(size=10, seed=42)
    ids1 = [item.image_id for item in s]
    ids2 = [item.image_id for item in source.CocoSource(size=10, seed=42)]
    assert ids1 == ids2               # 同种子可复现
    assert len(ids1) == 10


def test_coco_item_has_bytes_and_gt():
    s = source.CocoSource(size=1, seed=42)
    item = next(iter(s))
    assert isinstance(item.image_bytes, bytes) and len(item.image_bytes) > 0
    assert "categories" in item.gt      # gt 含该图 COCO 类别名列表
    assert "captions" in item.gt        # gt 含该图人工 caption 列表
```

标注: 该测试需真实 COCO 文件存在(config 已指向 val2014)。若数据缺失应 skip:
在测试顶部加:
```python
if not config.COCO_IMAGES.exists():
    pytest.skip("COCO 数据缺失", allow_module_level=True)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_source.py -v`
Expected: FAIL(`AttributeError: CocoSource`)

- [ ] **Step 3: 实现 source.py**

Create `taxo/backends/source.py`:
```python
"""ImageSource: 统一给出 (image_id, image_bytes, gt)。第一版实现 COCO。

换 CC3M 只需再写一个类实现同样的 __iter__ 契约(yield ImageItem)。
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path

from taxo import config


@dataclass
class ImageItem:
    image_id: str
    image_bytes: bytes
    gt: dict            # {"categories": [...], "captions": [...]}


class CocoSource:
    """从 COCO val 抽固定子集。gt = 该图的 80 类类别名 + 人工 captions。"""

    def __init__(self, size: int | None = None, seed: int | None = None):
        self.size = size or config.SUBSET_SIZE
        self.seed = seed if seed is not None else config.SUBSET_SEED
        self._build_index()

    def _build_index(self) -> None:
        inst = json.loads(config.COCO_INSTANCES.read_text("utf-8"))
        caps = json.loads(config.COCO_CAPTIONS.read_text("utf-8"))
        cat_name = {c["id"]: c["name"] for c in inst["categories"]}
        # image_id -> 文件名
        self._fname = {img["id"]: img["file_name"] for img in inst["images"]}
        # image_id -> set(类别名)
        self._cats: dict[int, set] = {}
        for a in inst["annotations"]:
            self._cats.setdefault(a["image_id"], set()).add(cat_name[a["category_id"]])
        # image_id -> [caption]
        self._caps: dict[int, list] = {}
        for a in caps["annotations"]:
            self._caps.setdefault(a["image_id"], []).append(a["caption"])
        # 固定子集: 只取有 caption 的图, 按种子抽样
        pool = sorted(self._caps.keys())
        rng = random.Random(self.seed)
        rng.shuffle(pool)
        self._subset = pool[: self.size]

    def __iter__(self):
        for iid in self._subset:
            fpath = config.COCO_IMAGES / self._fname[iid]
            if not fpath.exists():
                continue
            yield ImageItem(
                image_id=str(iid),
                image_bytes=fpath.read_bytes(),
                gt={"categories": sorted(self._cats.get(iid, set())),
                    "captions": self._caps.get(iid, [])},
            )

    def by_ids(self, ids: list[str]):
        """按 image_id 列表取子集(裂簇后只重抽碰撞图用)。"""
        want = set(ids)
        for item in self:
            if item.image_id in want:
                yield item
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_source.py -v`
Expected: 2 passed(或 skipped,若数据缺失)

- [ ] **Step 5: Commit**

```bash
git add prompt_lab/taxo/backends/source.py prompt_lab/taxo/tests/test_source.py
git commit -m "feat(taxo): COCO ImageSource + 测试"
```

---

## Task 6: backends/judge.py — Opus 4.8 裸 HTTP 客户端 + 缓存

**Files:**
- Create: `taxo/backends/judge.py`
- Test: `taxo/tests/test_judge.py`

背景: Opus 走 `~/.claude/settings.json` 的 `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN`,
`/v1/messages` 接口(已验证: model `Opus 4.8` 返回 `claude-opus-4-8`)。用 urllib
裸 HTTP, 不依赖 anthropic SDK(未安装)。缓存键 = `(prompt + schema_ver + prompt_ver)` 的 sha1。

- [ ] **Step 1: 写失败测试(纯逻辑部分, 不打网络)**

Create `taxo/tests/test_judge.py`:
```python
import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pathlib import Path
from taxo.backends import judge


def test_load_settings_reads_base_and_token(monkeypatch, tmp_path):
    fake = tmp_path / "settings.json"
    fake.write_text(json.dumps({"env": {
        "ANTHROPIC_BASE_URL": " https://x.com ",
        "ANTHROPIC_AUTH_TOKEN": "tok123"}}), "utf-8")
    base, tok = judge._load_settings(fake)
    assert base == "https://x.com"          # 去空白
    assert tok == "tok123"


def test_cache_hit_skips_call(tmp_path):
    j = judge.Judge(cache_dir=tmp_path)
    calls = {"n": 0}
    def fake_call(prompt):
        calls["n"] += 1
        return "RESULT"
    assert j._cached("keyA", fake_call, "p") == "RESULT"
    assert j._cached("keyA", fake_call, "p") == "RESULT"   # 第二次命中缓存
    assert calls["n"] == 1


def test_extract_json_from_fenced_block():
    txt = 'noise\n```json\n{"a": 1}\n```\ntrailing'
    assert judge.extract_json(txt) == {"a": 1}


def test_extract_json_bare():
    assert judge.extract_json('{"b": 2}') == {"b": 2}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_judge.py -v`
Expected: FAIL(`AttributeError`)

- [ ] **Step 3: 实现 judge.py**

Create `taxo/backends/judge.py`:
```python
"""Judge: Opus 4.8。三职责——裂簇提新 Key / ontology 合并判定 / metric 打分。

走 ~/.claude/settings.json 的 Anthropic 兼容端点, urllib 裸 HTTP。
缓存: 按内容 sha1 落盘, 续跑/重跑不重复烧钱。
"""
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from taxo import config

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_BARE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _load_settings(path: Path) -> tuple[str, str]:
    env = json.loads(Path(path).read_text("utf-8"))["env"]
    return env["ANTHROPIC_BASE_URL"].strip(), env["ANTHROPIC_AUTH_TOKEN"].strip()


def extract_json(text: str):
    """从 LLM 回复里抠出 JSON(优先 ```json``` 围栏, 退化到裸括号)。"""
    m = _FENCE.search(text) or _BARE.search(text)
    if not m:
        raise ValueError(f"no JSON in response: {text[:120]}")
    return json.loads(m.group(1))


class Judge:
    def __init__(self, cache_dir: Path, model: str | None = None,
                 settings_path: Path | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or config.JUDGE_MODEL
        self.base, self.token = _load_settings(settings_path or config.CLAUDE_SETTINGS)

    # ── 缓存 ──────────────────────────────────────────────
    def _cache_file(self, key: str) -> Path:
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _cached(self, key: str, fn, *args):
        cf = self._cache_file(key)
        if cf.exists():
            return json.loads(cf.read_text("utf-8"))["result"]
        result = fn(*args)
        cf.write_text(json.dumps({"key": key, "result": result}, ensure_ascii=False), "utf-8")
        return result

    # ── 底层调用 ──────────────────────────────────────────
    def _call(self, prompt: str, max_tokens: int = 1500) -> str:
        payload = {"model": self.model, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(
            self.base.rstrip("/") + "/v1/messages",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "x-api-key": self.token,
                     "anthropic-version": "2023-06-01"})
        resp = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
        return "".join(b.get("text", "") for b in resp["content"])

    def ask_json(self, prompt: str, cache_key: str, max_tokens: int = 1500):
        """带缓存的 JSON 问答。cache_key 应含 schema_ver+prompt_ver 保证失效正确。"""
        return self._cached(cache_key, lambda p: extract_json(self._call(p, max_tokens)), prompt)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_judge.py -v`
Expected: 4 passed

- [ ] **Step 5: 真实端点冒烟(手动, 不进 CI)**

Run:
```bash
cd prompt_lab && python -c "
from taxo.backends import judge
import tempfile
j = judge.Judge(cache_dir=tempfile.mkdtemp())
print(j.ask_json('Return exactly this JSON and nothing else: {\"ok\": true}', cache_key='smoke'))
"
```
Expected: `{'ok': True}`

- [ ] **Step 6: Commit**

```bash
git add prompt_lab/taxo/backends/judge.py prompt_lab/taxo/tests/test_judge.py
git commit -m "feat(taxo): Judge Opus 裸 HTTP + 缓存 + JSON 抽取 + 测试"
```

---

## Task 7: backends/extractor.py — dspy 驱动 gemma vision 抽 caption+JSON

**Files:**
- Create: `taxo/backends/extractor.py`
- Test: `taxo/tests/test_extractor.py`

背景: gemma 是 vision 端点。dspy 3.2.1 支持图像(`dspy.Image`)。抽取器把当前
Schema 的 active_keys(id/name/desc/allowed_values)拼进 prompt, 要求返回
`{key_id: value}`。学生 LM 用 `prompt_lab/lab_lm.py` 的 gemma 端点(关思考模式)。

- [ ] **Step 1: 写失败测试(结构+prompt 组装, 不打网络)**

Create `taxo/tests/test_extractor.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from taxo.backends import extractor


def test_render_keys_block_lists_active_keys():
    keys = [
        {"id": "k_000", "name": "scene", "desc": "场景", "value_type": "enum",
         "allowed_values": ["indoor", "outdoor"]},
        {"id": "k_001", "name": "primary_object", "desc": "主体", "value_type": "open",
         "allowed_values": []},
    ]
    block = extractor.render_keys_block(keys)
    assert "k_000" in block and "scene" in block
    assert "indoor" in block and "outdoor" in block   # enum 值出现
    assert "k_001" in block


def test_parse_output_keeps_only_known_keys():
    keys = [{"id": "k_000"}, {"id": "k_001"}]
    raw = {"k_000": "outdoor", "k_999": "junk", "k_001": ""}
    out = extractor.parse_output(raw, keys)
    assert out == {"k_000": "outdoor", "k_001": ""}   # 丢弃未知 k_999
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_extractor.py -v`
Expected: FAIL(`AttributeError`)

- [ ] **Step 3: 实现 extractor.py**

Create `taxo/backends/extractor.py`:
```python
"""Extractor: 用 dspy 驱动 gemma vision, 按当前 Schema 抽 caption + JSON。

gemma 端点复用 lab_lm.make_lm("gemma", ...) (已关思考模式)。
被 dspy 优化的对象是 ExtractBySchema 的 instructions。
"""
import base64
import json
import sys
from pathlib import Path

import dspy

# 复用 prompt_lab 根的 lab_lm(gemma/qwen 接线, 已踩坑)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from lab_lm import make_lm  # noqa: E402


def render_keys_block(keys: list[dict]) -> str:
    """把 active_keys 渲染成给 VLM 看的说明清单。"""
    lines = []
    for k in keys:
        vt = k.get("value_type", "open")
        allowed = k.get("allowed_values") or []
        hint = f" 允许值: {allowed}" if vt == "enum" and allowed else ""
        lines.append(f"- {k['id']} ({k.get('name','')}, {vt}): {k.get('desc','')}{hint}")
    return "\n".join(lines)


def parse_output(raw_json: dict, keys: list[dict]) -> dict:
    """只保留 Schema 里存在的 key_id, 丢弃 VLM 幻觉出的多余键。"""
    known = {k["id"] for k in keys}
    return {kid: v for kid, v in raw_json.items() if kid in known}


class ExtractBySchema(dspy.Signature):
    """看图, 先客观描述(caption), 再按给定 Key 清单抽取属性值。
    只输出清单里的 key_id; 图中没有的留空字符串; enum 值必须取自允许值。
    json_out 必须是合法 JSON 对象 {key_id: value}。"""
    image: dspy.Image = dspy.InputField(desc="待分析图像")
    keys_block: str = dspy.InputField(desc="Key 清单(id/名称/类型/描述/允许值)")
    caption: str = dspy.OutputField(desc="对图像的一句客观描述")
    json_out: str = dspy.OutputField(desc='JSON 对象, 形如 {"k_000":"outdoor"}')


class Extractor:
    def __init__(self, port: int = 8001, prompt_version: str = "v0"):
        self.lm = make_lm("gemma", port=port, max_tokens=1024, cache=False)
        self.prompt_version = prompt_version
        self.program = dspy.Predict(ExtractBySchema)

    def extract(self, image_bytes: bytes, keys: list[dict]) -> tuple[str, dict]:
        img = dspy.Image(url="data:image/jpeg;base64," +
                         base64.b64encode(image_bytes).decode())
        block = render_keys_block(keys)
        with dspy.context(lm=self.lm):
            pred = self.program(image=img, keys_block=block)
        try:
            raw = json.loads(pred.json_out)
        except (json.JSONDecodeError, TypeError):
            raw = {}
        return pred.caption, parse_output(raw, keys)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_extractor.py -v`
Expected: 2 passed

- [ ] **Step 5: 真实端点冒烟(手动)**

Run:
```bash
cd prompt_lab && python -c "
from taxo.backends.extractor import Extractor
from taxo.backends.source import CocoSource
e = Extractor()
item = next(iter(CocoSource(size=1, seed=42)))
keys = [{'id':'k_000','name':'scene','desc':'场景 indoor/outdoor','value_type':'enum','allowed_values':['indoor','outdoor']},
        {'id':'k_001','name':'primary_object','desc':'画面主体物体','value_type':'open','allowed_values':[]}]
cap, js = e.extract(item.image_bytes, keys)
print('caption:', cap)
print('json:', js)
"
```
Expected: 打印一句 caption + 形如 `{'k_000': 'outdoor', 'k_001': '...'}` 的 JSON。

- [ ] **Step 6: Commit**

```bash
git add prompt_lab/taxo/backends/extractor.py prompt_lab/taxo/tests/test_extractor.py
git commit -m "feat(taxo): dspy 驱动 gemma vision 抽取器 + 测试"
```

---

## Task 8: metrics.py — 四分量无监督 metric + 轮次指标聚合

**Files:**
- Create: `taxo/metrics.py`
- Test: `taxo/tests/test_metrics.py`

背景: 四分量 = stability(同图重抽 Jaccard) / validity(JSON 合法+enum 合规) /
coverage(非空占比) / faithfulness(Opus 1–5 分归一)。轮次指标 = distinctness /
collision_rate / new_key_yield / drop_rate。faithfulness 由 judge 提供,
这里只做纯计算, judge 分作为入参传入(便于单测)。

- [ ] **Step 1: 写失败测试**

Create `taxo/tests/test_metrics.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from taxo import metrics


def test_jaccard_identical_is_one():
    assert metrics.jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_is_zero():
    assert metrics.jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_both_empty_is_one():
    assert metrics.jaccard(set(), set()) == 1.0


def test_validity_all_valid():
    keys = [{"id": "k_0", "value_type": "enum", "allowed_values": ["a", "b"]}]
    assert metrics.validity({"k_0": "a"}, keys) == 1.0


def test_validity_penalizes_out_of_enum():
    keys = [{"id": "k_0", "value_type": "enum", "allowed_values": ["a", "b"]}]
    assert metrics.validity({"k_0": "zzz"}, keys) == 0.0


def test_coverage_counts_nonempty_ratio():
    keys = [{"id": "k_0"}, {"id": "k_1"}]
    assert metrics.coverage({"k_0": "x", "k_1": ""}, keys) == 0.5


def test_distinctness_from_clusters():
    # 5 张图, 一个 size=2 的碰撞簇 → 2 张碰撞 → distinctness = 1 - 2/5 = 0.6
    assert metrics.distinctness(n_images=5, clusters=[{"image_ids": ["a", "b"]}]) == 0.6


def test_new_key_yield():
    assert metrics.new_key_yield(n_new_keys=2, n_clusters=4) == 0.5
    assert metrics.new_key_yield(n_new_keys=1, n_clusters=0) == 0.0


def test_combine_weights():
    parts = {"stability": 1.0, "validity": 1.0, "coverage": 0.0, "faithfulness": 0.0}
    w = {"stability": 0.25, "validity": 0.25, "coverage": 0.25, "faithfulness": 0.25}
    assert metrics.combine(parts, w) == 0.5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_metrics.py -v`
Expected: FAIL(`AttributeError`)

- [ ] **Step 3: 实现 metrics.py**

Create `taxo/metrics.py`:
```python
"""无监督 metric 四分量 + 轮次指标。纯计算, faithfulness 分由外部(judge)传入。"""


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def stability(label_sets: list[set]) -> float:
    """同一图多次抽取的 label_set 两两 Jaccard 均值。<2 次视为满分。"""
    if len(label_sets) < 2:
        return 1.0
    pairs, total = 0, 0.0
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            total += jaccard(label_sets[i], label_sets[j])
            pairs += 1
    return total / pairs


def validity(json_canon: dict, keys: list[dict]) -> float:
    """enum 值须在 allowed_values 内; 合规键占(非空键)比例。全空视为满分。"""
    kmap = {k["id"]: k for k in keys}
    checked = [(kid, v) for kid, v in json_canon.items() if v not in ("", None)]
    if not checked:
        return 1.0
    ok = 0
    for kid, v in checked:
        k = kmap.get(kid)
        if k and k.get("value_type") == "enum" and k.get("allowed_values"):
            ok += 1 if v in k["allowed_values"] else 0
        else:
            ok += 1
    return ok / len(checked)


def coverage(json_canon: dict, keys: list[dict]) -> float:
    """非空 Value 数 / active Key 数。"""
    if not keys:
        return 0.0
    nonempty = sum(1 for v in json_canon.values() if v not in ("", None))
    return nonempty / len(keys)


def combine(parts: dict, weights: dict) -> float:
    """加权组合四分量。faithfulness 已归一到 0~1。"""
    return sum(parts[k] * weights[k] for k in weights)


# ── 轮次指标 ──────────────────────────────────────────────
def collision_image_count(clusters: list[dict]) -> int:
    return sum(len(c["image_ids"]) for c in clusters)


def distinctness(n_images: int, clusters: list[dict]) -> float:
    if n_images == 0:
        return 1.0
    return 1 - collision_image_count(clusters) / n_images


def collision_rate(n_images: int, clusters: list[dict]) -> float:
    if n_images == 0:
        return 0.0
    return collision_image_count(clusters) / n_images


def new_key_yield(n_new_keys: int, n_clusters: int) -> float:
    if n_clusters == 0:
        return 0.0
    return n_new_keys / n_clusters


def drop_rate(prev_clusters: int, cur_clusters: int) -> float:
    """碰撞簇数下降率。prev=0 时返回 0(无从下降)。"""
    if prev_clusters == 0:
        return 0.0
    return (prev_clusters - cur_clusters) / prev_clusters
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_metrics.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add prompt_lab/taxo/metrics.py prompt_lab/taxo/tests/test_metrics.py
git commit -m "feat(taxo): 四分量无监督 metric + 轮次指标 + 测试"
```

---

## Task 9: judge 的三个 prompt 方法(裂簇 / 合并 / 立规则)

**Files:**
- Modify: `taxo/backends/judge.py`(追加 `seed_schema` / `split_cluster` / `merge_decision` / `faithfulness` 四方法)
- Test: `taxo/tests/test_judge_prompts.py`

背景: 这四个方法都通过 `ask_json` 走缓存。测试用 monkeypatch 替换 `ask_json`
验证 prompt 组装与返回解析, **不打真实网络**。真实调用留冒烟步。

- [ ] **Step 1: 写失败测试**

Create `taxo/tests/test_judge_prompts.py`:
```python
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from taxo.backends import judge


def _judge(monkeypatch, canned):
    j = judge.Judge.__new__(judge.Judge)   # 跳过 __init__ 的 settings 读取
    j.captured = {}
    def fake_ask(prompt, cache_key, max_tokens=1500):
        j.captured["prompt"] = prompt
        j.captured["cache_key"] = cache_key
        return canned
    j.ask_json = fake_ask
    return j


def test_seed_schema_returns_key_list(monkeypatch):
    j = _judge(monkeypatch, {"keys": [
        {"name": "scene", "desc": "场景", "value_type": "enum",
         "allowed_values": ["indoor", "outdoor"]}]})
    out = j.seed_schema(base_prompt="场景/主体/动作", sample_captions=["a dog"])
    assert out[0]["name"] == "scene"
    assert "场景/主体/动作" in j.captured["prompt"]


def test_split_cluster_returns_new_keys(monkeypatch):
    j = _judge(monkeypatch, {"new_keys": [
        {"name": "dog_color", "desc": "狗的颜色", "value_type": "open",
         "allowed_values": []}]})
    out = j.split_cluster(cluster_captions=["black dog", "white dog"],
                          existing_keys=[{"id": "k_0", "name": "obj"}],
                          schema_ver=3, cluster_id="c5")
    assert out[0]["name"] == "dog_color"
    assert "c5" in j.captured["cache_key"] and "3" in j.captured["cache_key"]


def test_merge_decision_returns_verdict(monkeypatch):
    j = _judge(monkeypatch, {"decision": "merge", "into": "k_2"})
    out = j.merge_decision(new_key={"name": "canine"},
                           existing_keys=[{"id": "k_2", "name": "dog"}],
                           schema_ver=3)
    assert out["decision"] == "merge" and out["into"] == "k_2"


def test_faithfulness_returns_score(monkeypatch):
    j = _judge(monkeypatch, {"score": 4})
    s = j.faithfulness(caption="a dog", json_canon={"k_0": "dog"},
                       image_fp="fp123", schema_ver=3)
    assert s == 4
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_judge_prompts.py -v`
Expected: FAIL(`AttributeError: seed_schema`)

- [ ] **Step 3: 在 judge.py 追加四方法**

在 `taxo/backends/judge.py` 的 `Judge` 类**末尾**追加(与 `ask_json` 同缩进):
```python
    # ── 三职责 prompt ─────────────────────────────────────
    def seed_schema(self, base_prompt: str, sample_captions: list[str]) -> list[dict]:
        """Opus 先立规则: 基于基础 prompt + 样例 caption 生成初始 Key 集。"""
        prompt = (
            "你是视觉本体设计者。基于以下维度提示和样例图片描述, "
            "设计一组用于区分图像的属性 Key(8~15 个)。\n"
            f"维度提示: {base_prompt}\n"
            f"样例描述:\n" + "\n".join(f"- {c}" for c in sample_captions[:30]) + "\n\n"
            '只输出 JSON: {"keys":[{"name":..,"desc":..,"value_type":"enum|open|numeric|bool",'
            '"allowed_values":[..]}]}。enum 才填 allowed_values, 否则空数组。')
        return self.ask_json(prompt, cache_key=f"seed::{base_prompt}", max_tokens=2000)["keys"]

    def split_cluster(self, cluster_captions: list[str], existing_keys: list[dict],
                      schema_ver: int, cluster_id: str) -> list[dict]:
        """裂簇: 这些图 label 相同, 提议能分开它们的 1~3 个新 Key。"""
        ek = ", ".join(f"{k['id']}({k.get('name','')})" for k in existing_keys)
        prompt = (
            "以下图片当前标签完全相同, 但它们其实不同。请提议 1~3 个新属性 Key, "
            "使它们能被区分开。不要与已有 Key 重复。\n"
            f"已有 Key: {ek}\n"
            f"这些图的描述:\n" + "\n".join(f"- {c}" for c in cluster_captions[:20]) + "\n\n"
            '只输出 JSON: {"new_keys":[{"name":..,"desc":..,"value_type":..,"allowed_values":[..]}]}')
        return self.ask_json(
            prompt, cache_key=f"split::{cluster_id}::sv{schema_ver}", max_tokens=1200)["new_keys"]

    def merge_decision(self, new_key: dict, existing_keys: list[dict],
                       schema_ver: int) -> dict:
        """ontology 沉淀: 判断新 Key 是否与现有 Key 同义/上下位, 该不该合并。"""
        ek = "\n".join(f"- {k['id']}: {k.get('name','')} — {k.get('desc','')}"
                       for k in existing_keys)
        prompt = (
            f"新提议的 Key: {new_key.get('name')} — {new_key.get('desc','')}\n"
            f"已有 Key:\n{ek}\n\n"
            "判断: 新 Key 是否与某个已有 Key 同义(应合并)? 只输出 JSON: "
            '{"decision":"add|merge","into":"<被合并到的 key_id, add 时为 null>"}')
        return self.ask_json(
            prompt, cache_key=f"merge::{new_key.get('name')}::sv{schema_ver}", max_tokens=400)

    def faithfulness(self, caption: str, json_canon: dict, image_fp: str,
                     schema_ver: int) -> int:
        """metric 判官: 抽取忠实度 1~5(无幻觉/无冗余/覆盖到位)。"""
        prompt = (
            "评估以下图像属性抽取的忠实度(1~5): 5=完全忠实无幻觉, 1=严重幻觉/错误。\n"
            f"图像客观描述: {caption}\n"
            f"抽取结果: {json_canon}\n"
            '只输出 JSON: {"score": <1-5 整数>}')
        r = self.ask_json(prompt, cache_key=f"faith::{image_fp}::sv{schema_ver}", max_tokens=200)
        return int(r["score"])
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_judge_prompts.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add prompt_lab/taxo/backends/judge.py prompt_lab/taxo/tests/test_judge_prompts.py
git commit -m "feat(taxo): judge 立规则/裂簇/合并/忠实度四 prompt + 测试"
```

---

## Task 10: backends/reviewer.py — 三区 HTML review 页 + 可选门

**Files:**
- Create: `taxo/backends/reviewer.py`
- Test: `taxo/tests/test_reviewer.py`

背景: 仿 `videos/tools/vlm_preview.py` 的自包含 index.html(缩略图 base64 内嵌)。
三区: Schema 变更 / 碰撞簇 / 样本抽查。可选门 = 读 `round_XX/review.json`。

- [ ] **Step 1: 写失败测试**

Create `taxo/tests/test_reviewer.py`:
```python
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pathlib import Path
from taxo.backends import reviewer


def test_render_html_contains_three_sections():
    html = reviewer.render_html(
        round_no=2,
        new_keys=[{"id": "k_5", "name": "dog_color", "desc": "狗色", "reason": "拆黑白狗"}],
        clusters=[{"image_ids": ["1", "2"], "captions": ["black dog", "white dog"],
                   "thumbs_b64": ["", ""], "suggestion": "加 dog_color"}],
        samples=[{"image_id": "9", "caption": "a cat", "json": {"k_0": "cat"}, "thumb_b64": ""}],
        metrics={"distinctness": 0.6, "n_collision_clusters": 1})
    assert "Schema 变更" in html
    assert "碰撞簇" in html
    assert "样本抽查" in html
    assert "dog_color" in html
    assert "0.6" in html


def test_load_review_returns_none_when_absent():
    with tempfile.TemporaryDirectory() as tmp:
        assert reviewer.load_review(Path(tmp) / "review.json") is None


def test_load_review_reads_feedback():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "review.json"
        f.write_text('{"rejected_keys": ["k_5"], "renamed": {}}', "utf-8")
        fb = reviewer.load_review(f)
        assert fb["rejected_keys"] == ["k_5"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_reviewer.py -v`
Expected: FAIL(`AttributeError`)

- [ ] **Step 3: 实现 reviewer.py**

Create `taxo/backends/reviewer.py`:
```python
"""Reviewer: 每轮产自包含 index.html(缩略图 base64 内嵌) + 可选人工门。"""
import html as _html
import json
from pathlib import Path


def _esc(s) -> str:
    return _html.escape(str(s))


def _img_tag(b64: str) -> str:
    if not b64:
        return "<div class='noimg'>no image</div>"
    return f"<img src='data:image/jpeg;base64,{b64}'/>"


def render_html(round_no: int, new_keys: list[dict], clusters: list[dict],
                samples: list[dict], metrics: dict) -> str:
    """三区 HTML。new_keys/clusters/samples 均已含渲染所需字段。"""
    m = " | ".join(f"{k}={v}" for k, v in metrics.items())

    key_rows = "".join(
        f"<tr><td>{_esc(k['id'])}</td><td>{_esc(k['name'])}</td>"
        f"<td>{_esc(k.get('desc',''))}</td><td>{_esc(k.get('reason',''))}</td></tr>"
        for k in new_keys) or "<tr><td colspan=4>本轮无新增 Key</td></tr>"

    clus_cards = ""
    for c in clusters:
        thumbs = "".join(_img_tag(b) for b in c.get("thumbs_b64", []))
        caps = "<br>".join(_esc(x) for x in c.get("captions", []))
        clus_cards += (
            f"<div class='card'><div class='thumbs'>{thumbs}</div>"
            f"<div class='meta'><b>images:</b> {_esc(c['image_ids'])}<br>"
            f"<b>captions:</b><br>{caps}<br>"
            f"<b>Opus 建议:</b> {_esc(c.get('suggestion',''))}</div></div>")
    clus_cards = clus_cards or "<p>本轮无未解开碰撞簇</p>"

    samp_cards = "".join(
        f"<div class='card'>{_img_tag(s.get('thumb_b64',''))}"
        f"<div class='meta'>{_esc(s.get('caption',''))}<br>"
        f"<code>{_esc(json.dumps(s.get('json',{}), ensure_ascii=False))}</code></div></div>"
        for s in samples) or "<p>无抽查样本</p>"

    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>taxo round {round_no}</title><style>
body{{font-family:sans-serif;margin:20px;background:#f6f6f6}}
h2{{border-bottom:2px solid #888}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}
.card{{display:inline-block;vertical-align:top;background:#fff;border:1px solid #ddd;
margin:6px;padding:6px;max-width:340px}}
.thumbs img,.card>img{{max-height:120px;margin:2px}}
.noimg{{width:120px;height:80px;background:#eee;display:inline-block;text-align:center;line-height:80px;color:#999}}
.meta{{font-size:12px;margin-top:4px}}
code{{font-size:11px;color:#036}}
.metrics{{background:#eef;padding:8px;font-family:monospace}}
</style></head><body>
<h1>Taxonomy Discovery — Round {round_no}</h1>
<div class='metrics'>{_esc(m)}</div>
<h2>① Schema 变更</h2>
<table><tr><th>id</th><th>name</th><th>desc</th><th>Opus 理由</th></tr>{key_rows}</table>
<h2>② 碰撞簇(未解开)</h2>{clus_cards}
<h2>③ 样本抽查</h2>{samp_cards}
</body></html>"""


def write_html(path: Path, **kw) -> None:
    Path(path).write_text(render_html(**kw), "utf-8")


def load_review(path: Path):
    """读人工反馈; 不存在返回 None。结构: {rejected_keys:[], renamed:{id:newname}}。"""
    p = Path(path)
    return json.loads(p.read_text("utf-8")) if p.exists() else None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_reviewer.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add prompt_lab/taxo/backends/reviewer.py prompt_lab/taxo/tests/test_reviewer.py
git commit -m "feat(taxo): 三区 HTML review 页 + 可选门 + 测试"
```

---

## Task 11: run_round.py — 单轮编排

**Files:**
- Create: `taxo/run_round.py`
- Test: `taxo/tests/test_run_round.py`(用 fake 后端, 不打网络)

背景: 编排 抽取→归一化→碰撞→裂簇→沉淀→产 HTML。为可测,
`run_round(ctx)` 接收一个含各后端的 ctx 对象;测试传 fake。

- [ ] **Step 1: 写失败测试**

Create `taxo/tests/test_run_round.py`:
```python
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pathlib import Path
from types import SimpleNamespace
from taxo import run_round
from taxo.core import schema as schema_mod


class FakeSource:
    def __init__(self, items): self.items = items
    def __iter__(self): return iter(self.items)
    def by_ids(self, ids):
        return [i for i in self.items if i.image_id in set(ids)]


def _item(iid): return SimpleNamespace(image_id=iid, image_bytes=b"x",
                                       gt={"categories": [], "captions": []})


def test_round_produces_records_and_collisions(tmp_path):
    reg = schema_mod.SchemaRegistry(tmp_path)
    reg.add_key(name="obj", desc="", value_type="open",
                introduced_round=0, introduced_by="seed")
    reg.snapshot()
    # 两张图抽出相同 label → 必碰撞
    fake_extract = lambda b, keys: ("a dog", {"k_000": "dog"})
    ctx = SimpleNamespace(
        source=FakeSource([_item("1"), _item("2")]),
        registry=reg,
        canon_map={},
        extract_fn=fake_extract,
        round_dir=tmp_path / "rounds" / "round_00",
        round_no=0,
        participant_ids=None,          # None = 全体
    )
    result = run_round.run_round(ctx)
    assert result["n_images"] == 2
    assert len(result["clusters"]) == 1
    assert set(result["clusters"][0]["image_ids"]) == {"1", "2"}
    # 记录已落盘
    from taxo.core import record
    assert len(record.read_all(ctx.round_dir / "records.jsonl")) == 2


def test_round_no_collision_when_labels_differ(tmp_path):
    reg = schema_mod.SchemaRegistry(tmp_path)
    reg.add_key(name="obj", desc="", value_type="open",
                introduced_round=0, introduced_by="seed")
    reg.snapshot()
    seq = iter([("a dog", {"k_000": "dog"}), ("a cat", {"k_000": "cat"})])
    ctx = SimpleNamespace(
        source=FakeSource([_item("1"), _item("2")]),
        registry=reg, canon_map={},
        extract_fn=lambda b, keys: next(seq),
        round_dir=tmp_path / "rounds" / "round_00",
        round_no=0, participant_ids=None)
    result = run_round.run_round(ctx)
    assert result["clusters"] == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_run_round.py -v`
Expected: FAIL(`AttributeError: run_round`)

- [ ] **Step 3: 实现 run_round.py**

Create `taxo/run_round.py`:
```python
"""单轮编排: 抽取 → 归一化 → 碰撞检测 → 落盘 → 返回本轮结果。

裂簇/沉淀/HTML 由 loop.py 在拿到 result 后调 judge/reviewer 完成——
run_round 只负责"抽取+碰撞"这一确定性部分, 便于单测。
"""
import time
from taxo.core import canon, collide, record


def run_round(ctx) -> dict:
    """ctx 需含: source, registry, canon_map, extract_fn, round_dir, round_no,
    participant_ids(None=全体, 否则只抽这些 image_id)。
    返回: {n_images, records, clusters}。
    """
    keys = ctx.registry.active_keys()
    records_path = ctx.round_dir / "records.jsonl"
    ctx.round_dir.mkdir(parents=True, exist_ok=True)

    if ctx.participant_ids is None:
        items = list(ctx.source)
    else:
        items = list(ctx.source.by_ids(ctx.participant_ids))

    done = record.done_ids(records_path)   # 续跑: 跳过已抽的
    rows = []
    for item in items:
        if item.image_id in done:
            continue
        caption, json_raw = ctx.extract_fn(item.image_bytes, keys)
        json_canon = canon.canonicalize_json(json_raw, ctx.canon_map)
        fp = collide.fingerprint(json_canon)
        row = {
            "image_id": item.image_id, "round": ctx.round_no,
            "caption": caption, "json_raw": json_raw, "json_canon": json_canon,
            "label_set_fp": fp,
            "label_set": [f"{k}={v}" for k, v in collide.label_pairs(json_canon)],
            "extractor": {"ts": time.time(), "schema_ver": ctx.registry.version},
        }
        record.append(records_path, row)
        rows.append(row)

    all_rows = record.read_all(records_path)
    clusters = collide.find_collisions(all_rows)
    return {"n_images": len(all_rows), "records": all_rows, "clusters": clusters}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_run_round.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add prompt_lab/taxo/run_round.py prompt_lab/taxo/tests/test_run_round.py
git commit -m "feat(taxo): 单轮编排(抽取+归一化+碰撞) + 测试"
```

---

## Task 12: loop.py — 双闸终止判据 + 多轮驱动

**Files:**
- Create: `taxo/loop.py`
- Test: `taxo/tests/test_loop.py`(只测终止逻辑, 纯函数)

背景: `should_stop` 是纯函数, 单测覆盖三闸。多轮驱动 `run_loop` 编排
run_round + judge 裂簇/沉淀 + reviewer, 走真实后端(手动冒烟), 不进单测。

- [ ] **Step 1: 写失败测试(只测终止判据)**

Create `taxo/tests/test_loop.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from taxo import loop


def test_stop_when_zero_collisions():
    stop, reason = loop.should_stop(
        history=[{"n_collision_clusters": 0, "n_keys_new": 0}],
        n_active_keys=10)
    assert stop and reason == "distinctness"


def test_stop_on_convergence_window():
    # 连续 2 轮: 新 Key<=1 且下降率<10%
    hist = [
        {"n_collision_clusters": 10, "n_keys_new": 0},
        {"n_collision_clusters": 10, "n_keys_new": 1},   # 下降率 0
        {"n_collision_clusters": 10, "n_keys_new": 0},   # 下降率 0
    ]
    stop, reason = loop.should_stop(history=hist, n_active_keys=10)
    assert stop and reason == "convergence"


def test_no_stop_when_still_dropping():
    hist = [
        {"n_collision_clusters": 20, "n_keys_new": 3},
        {"n_collision_clusters": 10, "n_keys_new": 2},   # 下降率 50%
    ]
    stop, _ = loop.should_stop(history=hist, n_active_keys=10)
    assert not stop


def test_stop_on_max_rounds():
    hist = [{"n_collision_clusters": 5, "n_keys_new": 2}] * loop_max()
    stop, reason = loop.should_stop(history=hist, n_active_keys=10)
    assert stop and reason == "max_rounds"


def test_stop_on_key_limit():
    stop, reason = loop.should_stop(
        history=[{"n_collision_clusters": 5, "n_keys_new": 2}],
        n_active_keys=999)
    assert stop and reason == "key_limit"


def loop_max():
    from taxo import config
    return config.MAX_ROUNDS
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_loop.py -v`
Expected: FAIL(`AttributeError: should_stop`)

- [ ] **Step 3: 实现 loop.py(先实现 should_stop, run_loop 编排)**

Create `taxo/loop.py`:
```python
"""多轮驱动 + 双闸终止。should_stop 纯函数可单测; run_loop 编排真实后端。"""
from taxo import config


def should_stop(history: list[dict], n_active_keys: int) -> tuple[bool, str]:
    """双闸 + 安全阀。history: 每轮 metrics(至少含 n_collision_clusters/n_keys_new)。
    返回 (是否停, 原因)。判定优先级: 区分性 > key上限 > max轮 > 收敛。
    """
    last = history[-1]
    # 闸①: 区分性达标
    if last["n_collision_clusters"] == 0:
        return True, "distinctness"
    # 闸③安全阀: Key 上限
    if n_active_keys >= config.MAX_KEYS:
        return True, "key_limit"
    # 闸③安全阀: 最大轮次
    if len(history) >= config.MAX_ROUNDS:
        return True, "max_rounds"
    # 闸②: 收敛(连续 CONVERGE_WINDOW 轮 新Key<=阈值 且 下降率<阈值)
    win = config.CONVERGE_WINDOW
    if len(history) >= win + 1:                    # 需前一轮算下降率
        ok = True
        for i in range(len(history) - win, len(history)):
            cur, prev = history[i], history[i - 1]
            drop = 0.0 if prev["n_collision_clusters"] == 0 else \
                (prev["n_collision_clusters"] - cur["n_collision_clusters"]) \
                / prev["n_collision_clusters"]
            if not (cur["n_keys_new"] <= config.CONVERGE_MAX_NEW_KEYS
                    and drop < config.CONVERGE_MIN_DROP_RATE):
                ok = False
                break
        if ok:
            return True, "convergence"
    return False, ""
```

标注: `run_loop` 编排函数在下个 Task 里加(它依赖 judge/reviewer 全就位),
本 Task 只交付可单测的 `should_stop`。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd prompt_lab && python -m pytest taxo/tests/test_loop.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add prompt_lab/taxo/loop.py prompt_lab/taxo/tests/test_loop.py
git commit -m "feat(taxo): 双闸终止判据 should_stop + 测试"
```

---

## Task 13: loop.run_loop — 端到端编排 + 续跑

**Files:**
- Modify: `taxo/loop.py`(追加 `run_loop` 与裂簇/沉淀/HTML 编排 + `main`)
- Modify: `taxo/backends/source.py` 无需改
- Test: 端到端冒烟(手动, 20 张图)

背景: 把所有单元串起来。第 1 轮全体抽取; 之后每轮取上轮碰撞簇图做 participant,
裂簇提新 Key → merge_decision 沉淀 → snapshot → 重抽碰撞图 → 产 HTML →
(可选)review 门 → should_stop。

- [ ] **Step 1: 追加 run_loop 到 loop.py**

在 `taxo/loop.py` 末尾追加:
```python
import base64
import json as _json
from pathlib import Path

from taxo import metrics, run_round
from taxo.core import canon, record, schema as schema_mod
from taxo.backends import reviewer


def _thumb_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode()


def _apply_review(registry, feedback):
    """人工反馈: rejected_keys 软删, renamed 改名。"""
    if not feedback:
        return
    for kid in feedback.get("rejected_keys", []):
        if registry.get(kid):
            registry.keys[kid]["synonyms_of"] = "__rejected__"
    for kid, newname in feedback.get("renamed", {}).items():
        if registry.get(kid):
            registry.keys[kid]["name"] = newname


def run_loop(source, registry, judge, run_dir: Path, base_prompt: str):
    """端到端: 立规则 → 多轮(抽取/碰撞/裂簇/沉淀/HTML/门) → 双闸停。"""
    run_dir = Path(run_dir)
    state = record.load_cursor(run_dir / "state.json", default={"last_round": -1})
    canon_map = canon.load_map(run_dir / "schema" / "canon_map.v0.json")

    # ── 第 0 步: Opus 立规则(仅 registry 为空时) ──
    if registry.n_active() == 0:
        sample_caps = []
        for i, item in enumerate(source):
            sample_caps.extend(item.gt.get("captions", [])[:1])
            if i >= 30:
                break
        for k in judge.seed_schema(base_prompt, sample_caps):
            registry.add_key(name=k["name"], desc=k.get("desc", ""),
                             value_type=k.get("value_type", "open"),
                             allowed_values=k.get("allowed_values", []),
                             introduced_round=0, introduced_by="seed")
        registry.snapshot()

    history = []
    participant_ids = None                 # 首轮全体
    round_no = state["last_round"] + 1
    # 缓存 image_bytes 供裂簇/HTML 用(小子集可全驻留)
    items_by_id = {it.image_id: it for it in source}

    while round_no < config.MAX_ROUNDS:
        round_dir = run_dir / "rounds" / f"round_{round_no:02d}"
        ctx = _mk_ctx(source, registry, canon_map, judge,
                      round_dir, round_no, participant_ids)
        result = run_round.run_round(ctx)
        clusters = result["clusters"]

        # 裂簇 + 沉淀
        new_keys_meta = []
        for ci, c in enumerate(clusters):
            caps = [items_by_id[i].gt.get("captions", [""])[0]
                    if i in items_by_id else "" for i in c["image_ids"]]
            for nk in judge.split_cluster(caps, registry.active_keys(),
                                          registry.version, f"r{round_no}c{ci}"):
                dec = judge.merge_decision(nk, registry.active_keys(), registry.version)
                if dec["decision"] == "add":
                    kid = registry.add_key(
                        name=nk["name"], desc=nk.get("desc", ""),
                        value_type=nk.get("value_type", "open"),
                        allowed_values=nk.get("allowed_values", []),
                        introduced_round=round_no, introduced_by=f"cluster#{ci}")
                    new_keys_meta.append({**registry.get(kid), "reason": nk.get("desc", "")})
        if new_keys_meta:
            registry.snapshot()

        # 指标
        m = {
            "round": round_no, "n_images": result["n_images"],
            "n_keys_total": registry.n_active(), "n_keys_new": len(new_keys_meta),
            "n_collision_clusters": len(clusters),
            "max_cluster_size": max((len(c["image_ids"]) for c in clusters), default=0),
            "collision_rate": round(metrics.collision_rate(result["n_images"], clusters), 4),
            "distinctness": round(metrics.distinctness(result["n_images"], clusters), 4),
            "new_key_yield": round(metrics.new_key_yield(len(new_keys_meta), len(clusters)), 4),
        }
        (round_dir / "metrics.json").write_text(
            _json.dumps(m, ensure_ascii=False, indent=2), "utf-8")
        history.append(m)

        # HTML review 页
        clus_view = [{
            "image_ids": c["image_ids"],
            "captions": [items_by_id[i].gt.get("captions", [""])[0]
                         if i in items_by_id else "" for i in c["image_ids"][:6]],
            "thumbs_b64": [_thumb_b64(items_by_id[i].image_bytes)
                           if i in items_by_id else "" for i in c["image_ids"][:6]],
            "suggestion": ""} for c in clusters[:20]]
        samples = [{"image_id": r["image_id"], "caption": r["caption"],
                    "json": r["json_canon"],
                    "thumb_b64": _thumb_b64(items_by_id[r["image_id"]].image_bytes)
                    if r["image_id"] in items_by_id else ""}
                   for r in result["records"][:12]]
        reviewer.write_html(round_dir / "index.html", round_no=round_no,
                            new_keys=new_keys_meta, clusters=clus_view,
                            samples=samples, metrics=m)

        # 可选 review 门
        if config.REVIEW_MODE == "on":
            fb = _wait_review(round_dir / "review.json")
            _apply_review(registry, fb)
            if fb:
                registry.snapshot()

        # 存游标
        record.save_cursor(run_dir / "state.json", {"last_round": round_no})

        # 双闸判停
        stop, reason = should_stop(history, registry.n_active())
        if stop:
            print(f"[loop] stop @ round {round_no}: {reason}")
            break

        # 下一轮: 只处理本轮碰撞图(incremental)
        participant_ids = None if config.COLLISION_SCOPE == "global" else \
            [i for c in clusters for i in c["image_ids"]]
        round_no += 1

    return history


def _mk_ctx(source, registry, canon_map, judge, round_dir, round_no, participant_ids):
    from types import SimpleNamespace
    return SimpleNamespace(
        source=source, registry=registry, canon_map=canon_map,
        extract_fn=judge_extract_fn(), round_dir=round_dir,
        round_no=round_no, participant_ids=participant_ids)


# extract_fn 由 main 注入真实 Extractor; 这里留个占位在 main 覆盖
_EXTRACT_FN = None
def judge_extract_fn():
    return _EXTRACT_FN


def _wait_review(path: Path):
    """REVIEW_MODE=on: 等 review.json 出现。TIMEOUT=0 表示不等待(跳过)。"""
    import time
    if config.REVIEW_TIMEOUT_S <= 0:
        return reviewer.load_review(path)
    waited = 0
    while waited < config.REVIEW_TIMEOUT_S:
        fb = reviewer.load_review(path)
        if fb is not None:
            return fb
        time.sleep(5)
        waited += 5
    return None
```

标注: `_EXTRACT_FN` 全局由 `main` 注入真实 `Extractor.extract`——保持 run_round
的 `extract_fn` 可注入(单测传 fake, 生产传 gemma)。

- [ ] **Step 2: 追加 main 入口**

在 `taxo/loop.py` 末尾继续追加:
```python
def main():
    import taxo.loop as L
    from taxo.backends.source import CocoSource
    from taxo.backends.extractor import Extractor
    from taxo.backends.judge import Judge

    run_dir = config.RUNS_DIR / "coco_proto"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(_json.dumps({
        "source": "coco", "split": config.COCO_SPLIT,
        "subset_size": config.SUBSET_SIZE, "seed": config.SUBSET_SEED,
        "scope": config.COLLISION_SCOPE}, ensure_ascii=False, indent=2), "utf-8")

    source = list(CocoSource())            # 驻留(小子集)
    class _Src:                            # 包装成可迭代 + by_ids
        def __iter__(self): return iter(source)
        def by_ids(self, ids):
            s = set(ids); return [i for i in source if i.image_id in s]

    registry = schema_mod.SchemaRegistry(run_dir)
    judge = Judge(cache_dir=run_dir / "judge_cache")
    extractor = Extractor()
    L._EXTRACT_FN = extractor.extract

    base_prompt = "场景 / 主体 / 动作 / 物体 / 空间关系 / 视角 / 构图"
    hist = run_loop(_Src(), registry, judge, run_dir, base_prompt)
    print(_json.dumps(hist, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 全量单测回归(确认没弄坏前面)**

Run: `cd prompt_lab && python -m pytest taxo/tests/ -v`
Expected: 全部 passed(source 测试可能 skipped)

- [ ] **Step 4: 端到端小冒烟(手动, 改小规模)**

Run:
```bash
cd prompt_lab && python -c "
from taxo import config
config.SUBSET_SIZE = 20      # 临时缩小
config.MAX_ROUNDS = 2
from taxo.loop import main
main()
"
```
Expected: 跑完 1~2 轮,`runs/coco_proto/rounds/round_00/index.html` 生成,
metrics.json 有 distinctness 数字,无异常栈。打开 HTML 应见三区。

- [ ] **Step 5: Commit**

```bash
git add prompt_lab/taxo/loop.py
git commit -m "feat(taxo): run_loop 端到端编排 + 续跑 + main 入口"
```

---

## Task 14: dspy GEPA 优化抽取器(懒优化, 可选启用)

**Files:**
- Create: `taxo/optimize.py`
- Test: 手动冒烟(GEPA 需真实端点)

背景: 落实"dspy 优化抽取器 prompt"。用 GEPA: 学生 gemma 跑抽取,
teacher Opus 反思重写 instructions,metric = 四分量组合(§8)。懒优化:
loop 里当轮新增 Key ≥ `REOPT_KEY_THRESHOLD` 才调。第一版先独立可跑,
接入 loop 留作后续(不阻塞闭环)。

- [ ] **Step 1: 实现 optimize.py**

Create `taxo/optimize.py`:
```python
"""GEPA 优化抽取器 instructions。学生 gemma 跑预测, teacher Opus 反思。

metric = stability/validity/coverage/faithfulness 加权(§8)。
用法: python -m taxo.optimize (需 gemma + Opus 端点在线)。
"""
import sys
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab_lm import make_lm  # noqa: E402

from taxo import config, metrics
from taxo.backends.extractor import ExtractBySchema, render_keys_block, parse_output
from taxo.backends.source import CocoSource
from taxo.backends.judge import Judge
from taxo.core import canon, collide


def build_metric(judge: Judge, keys: list[dict]):
    """返回 dspy metric: 无监督四分量 + Opus 忠实度。附文字反馈供 GEPA 反思。"""
    import json as _json
    w = config.METRIC_WEIGHTS

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        try:
            jr = parse_output(_json.loads(pred.json_out), keys)
        except Exception:
            return dspy.Prediction(score=0.0, feedback="json_out 非合法 JSON")
        jc = canon.canonicalize_json(jr, {})
        parts = {
            "stability": 1.0,   # 单次评测不算重抽, 置 1(重抽稳定性在 loop 层量)
            "validity": metrics.validity(jc, keys),
            "coverage": metrics.coverage(jc, keys),
            "faithfulness": judge.faithfulness(pred.caption, jc, gold.image_fp, 0) / 5.0,
        }
        score = metrics.combine(parts, w)
        fb = f"validity={parts['validity']:.2f} coverage={parts['coverage']:.2f} " \
             f"faithfulness={parts['faithfulness']:.2f}"
        return dspy.Prediction(score=score, feedback=fb)
    return metric


def main():
    student = make_lm("gemma", port=8001, cache=False)
    teacher = make_lm("gemma", port=8001, max_tokens=2048, cache=False)  # 可换 Opus 端点
    dspy.configure(lm=student)

    keys = [{"id": "k_000", "name": "scene", "desc": "室内/室外",
             "value_type": "enum", "allowed_values": ["indoor", "outdoor"]},
            {"id": "k_001", "name": "primary_object", "desc": "画面主体",
             "value_type": "open", "allowed_values": []}]
    block = render_keys_block(keys)

    import base64
    examples = []
    for item in list(CocoSource(size=12, seed=1)):
        img = dspy.Image(url="data:image/jpeg;base64," +
                         base64.b64encode(item.image_bytes).decode())
        examples.append(dspy.Example(
            image=img, keys_block=block, image_fp=item.image_id
        ).with_inputs("image", "keys_block"))
    trainset, valset = examples[:8], examples[8:]

    judge = Judge(cache_dir=config.RUNS_DIR / "opt_cache")
    program = dspy.Predict(ExtractBySchema)
    gepa = dspy.GEPA(metric=build_metric(judge, keys), reflection_lm=teacher,
                     max_metric_calls=40, num_threads=2, track_stats=True)
    optimized = gepa.compile(program, trainset=trainset, valset=valset)
    print("== 优化后 instructions ==")
    print(optimized.signature.instructions)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法/导入检查**

Run: `cd prompt_lab && python -c "import taxo.optimize; print('import ok')"`
Expected: `import ok`

- [ ] **Step 3: 手动冒烟(需端点在线, 可跳过)**

Run: `cd prompt_lab && python -m taxo.optimize`
Expected: 跑完打印优化后的 instructions(耗时数十秒)。

- [ ] **Step 4: Commit**

```bash
git add prompt_lab/taxo/optimize.py
git commit -m "feat(taxo): GEPA 抽取器优化(独立可跑, 懒接入 loop)"
```

---

## Task 15: README + 全量回归

**Files:**
- Create: `taxo/README.md`
- Test: 全量 pytest

- [ ] **Step 1: 写 README**

Create `taxo/README.md`:
```markdown
# taxo — 迭代式 Taxonomy 发现闭环

从图像子集(COCO 原型)自动发现一套能区分图像的属性 Key 体系(Schema)。
方案 A: VLM 抽 caption+JSON → label_set 碰撞检测 → Opus 裂簇提新 Key →
ontology 沉淀 → 双闸终止。设计见 `docs/superpowers/specs/2026-07-08-taxonomy-discovery-loop-design.md`。

## 环境
    source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
    cd .../sport_ontology/prompt_lab

## 跑
    python -m taxo.loop            # 端到端(需 gemma:8001 + Opus 端点)
    python -m taxo.optimize        # 单独跑 GEPA 优化抽取器

## 结构
- core/  纯函数: schema(版本化 Key) / canon(归一化) / collide(碰撞) / record(记录)
- backends/  source(COCO) / extractor(gemma) / judge(Opus) / reviewer(HTML)
- metrics.py 四分量无监督 metric
- run_round.py 单轮  |  loop.py 多轮+双闸终止  |  optimize.py GEPA

## 产物
`runs/<run_id>/`(gitignore): schema/vN.json, rounds/round_XX/{records.jsonl,
metrics.json, index.html}, state.json(续跑游标)。打开 index.html 看每轮 review。

## 关键参数(config.py)
SUBSET_SIZE / MAX_ROUNDS / MAX_KEYS / COLLISION_SCOPE(incremental|global) /
REVIEW_MODE(off|on) / METRIC_WEIGHTS。

## 测试
    python -m pytest taxo/tests/ -v
```

- [ ] **Step 2: 全量回归**

Run: `cd prompt_lab && python -m pytest taxo/tests/ -v`
Expected: 全部 passed(source 若无数据则 skipped)。

- [ ] **Step 3: Commit**

```bash
git add prompt_lab/taxo/README.md
git commit -m "docs(taxo): README + 全量测试回归"
```

---

## 完成标准

- 全量 `python -m pytest taxo/tests/ -v` 通过。
- `python -m taxo.loop`(SUBSET_SIZE 缩到 20 冒烟)能跑完 ≥1 轮,产出
  `rounds/round_00/index.html` + metrics.json,无异常。
- HTML 三区可见;Schema 快照 `schema/v*.json` 随轮增长;双闸能触发停止。
- 续跑: 中断后重跑不重抽已完成图(state.json 生效)。




