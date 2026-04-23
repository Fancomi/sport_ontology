# source /home/baidu/envs/mineru/bin/activate
# source /home/baidu/envs/swift/bin/activate

# ── 统一配置 ──────────────────────────────────────────────────────────────────
HOST="127.0.0.1"
# Gemma
PORT="8001,8002,8003,8004,8005,8006,8007,8008"
WORKERS=8

# Qwen3.6
PORT="8001,8002,8003,8004"
WORKERS=4

# ─────────────────────────────────────────────────────────────────────────────

# 视频描述 / 处理预热
# python video_frames.py --max-side 768
# python video_frames.py --prebuild --max-side 768

# # 1. 翻译metadata到metadata_cn
# python3 1_translate_wiki.py --host $HOST --port $PORT -w $WORKERS

# 2. VLM 扩写视频描述（--check 启用 LLM 质检，质检复用同一组端口）
# python3 2_augment_wiki.py --host $HOST --port $PORT --check -w $WORKERS

# 2.1 批量校验 augment_xxx.json 合规性 [已合入2混合调用]
# python3 2_1_check_augment.py --host $HOST --port $PORT -w $WORKERS

# # 3. 槽位统计: 覆盖输出 slot_vocab.json / slot_abnormal.json / slot_vocab.png
# python3 3_collect_slots.py [--top 20]

# # 4. (已删除) 4_fetch_vocab_info.py — Wordnet 信息收集，用途有限

# # 5. LLM 补充本体属性：读取 slot_vocab.json，产出/更新 slot_ontology.json
# #    清理过期键、补充新词、保留已有条目
# python3 5_enrich_with_llm.py --host $HOST --port $PORT -w $WORKERS

# # 5.1 基于 LLM 清理 slot_ontology.json 中不恰当的混淆关系。
# python3 5_1_clean_ontology.py --host $HOST --port $PORT -w $WORKERS

# # # 5.2 关系对称传播增强（无 LLM，纯集合运算，可反复运行直到 5_1 收敛）
# # #    默认输出到临时文件 slot_ontology_infer.json，确认后加 --output slot_ontology.json 覆盖
# # python3 5_2_infer_relations.py --output slot_ontology_temp.json
# # python3 5_2_infer_relations.py --output slot_ontology.json
# # # 5.2 完成后建议重跑 5.1 清理传播引入的噪声（循环直到两脚本均无变化）
# for i in $(seq 1 5); do
#     python3 5_1_clean_ontology.py --host $HOST --port $PORT -w $WORKERS
# done

# # 6. 图谱构建[可视化]
# python3 6_build_wiki.py

# # 8. VLM 评测（在线采样 confusable / 重刷 hard）
# #    --mode confusable  在线采样评测，结果追加 eval_results.jsonl
# #    --mode hard        重刷累计 hard 分数（先跑 9 --reset-counts）
# #    --mode all         全部（默认）
# python3 8_eval_confusable.py --host $HOST --port $PORT -w $WORKERS --mode confusable
# python3 8_eval_confusable.py --host $HOST --port $PORT -w $WORKERS --mode hard
# python3 8_eval_confusable.py --host $HOST --port $PORT -w $WORKERS

# # 8.3 完形填空评测（在线抽样，不依赖 step7 产物）
# #     --limit N  限制文件数（调试）；--dry-run 只看 prompt 不调 VLM
# python3 8_3_cloze_eval.py --host $HOST --port $PORT -w $WORKERS --dry-run --limit 4
# python3 8_3_cloze_eval.py --host $HOST --port $PORT -w $WORKERS --limit 4

# # 8.1 分析混淆判断结果
# python3 8_1_analyze.py \
#     --input eval_results_hard.jsonl \
#     --out   eval_accuracy_hard.png \
#     --stats eval_stats_hard.json
# python3 8_1_analyze.py --compare \
# BAKUP/eval_results_v2_gemma.jsonl \
# BAKUP/eval_results_v2_qwen3.6.jsonl \
# --labels Gemma Qwen3.6

# # 9. 从 eval_results.jsonl 提取答错对，幂等合入 hard_all.jsonl
# #    --input  可指定多个文件取并集
# #    --clean  清理 augment 更新后 [slot:orig] 已失效的历史条目
# #    --reset-counts  清零所有 error_count（在重新跑 step 8 --mode hard 前执行）
# python3 9_extract_errors.py --input eval_results.jsonl
# python3 9_extract_errors.py --reset-counts
# python3 9_extract_errors.py \
#     --input \
#     BAKUP/eval_results_v2_gemma.jsonl \
#     BAKUP/eval_results_v2_qwen3.6.jsonl \
#     --clean --reset-counts

# # 9.1 LLM 审核 hard_all.jsonl 句子级有效性（删除上下文等价 / 视觉不可辨条目）
# #     --dry-run 只看判断结果，不写文件；--verbose 打印完整 prompt（配合 --limit）
# python3 9_1_clean_hard.py --host $HOST --port $PORT -w $WORKERS --dry-run --verbose --limit 4
# python3 9_1_clean_hard.py --host $HOST --port $PORT -w $WORKERS

# # 9.2 渲染 hard_all.jsonl → hn_render_{view}.json（单向，供人工标注）
# #     输出到每个视频叶目录；hn_render 前缀表示渲染产物，区别于 hard_all 数据源
# #     兼容单槽替换型（confusable/incompatibility）和完形填空型（__cloze__）
# #     每条 pair 含 hard_key，标注完成后可按 key 写回 hard_all.jsonl
# #
# #     渲染全量（覆盖已有文件）
# python3 9_2_render_hard.py
# #     指定不同来源版本
# python3 9_2_render_hard.py --input BAKUP/hard_all_v2.jsonl
# #     只渲染正面视角
# python3 9_2_render_hard.py --views front
# #     删除所有渲染文件
# python3 9_2_render_hard.py --clean
