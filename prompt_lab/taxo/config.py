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
