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
