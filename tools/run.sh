# source /home/baidu/envs/mineru/bin/activate
# source /home/baidu/envs/swift/bin/activate

# ── 统一配置 ──────────────────────────────────────────────────────────────────
HOST="127.0.0.1"
PORT="8001,8002,8003,8004,8005,8006,8007,8008"
WORKERS=8
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

# # 6. 图谱构建[可视化]
# python3 6_build_wiki.py

# # 7. 生成混淆样本
# python3 7_gen_confusable.py

# # 8. 评测混淆判断 产出json
# #    --mode confusable  只评测 step7 新产物（confusable_xxx.json），结果追加 eval_results.jsonl
# #    --mode hard        只刷累计 hard 分数（hard_xxx.json），更新 error_count（先跑 9 --reset-counts）
# #    --mode all         全部（默认）
# python3 8_eval_confusable.py --host $HOST --port $PORT -w $WORKERS --mode confusable
# python3 8_eval_confusable.py --host $HOST --port $PORT -w $WORKERS --mode hard
# python3 8_eval_confusable.py --host $HOST --port $PORT -w $WORKERS

# # 8.1 分析混淆判断结果
# python3 8_1_analyze.py --compare \
# BAKUP/eval_results_v2_gemma.jsonl \
# BAKUP/eval_results_v2_qwen3.6.jsonl \
# --labels Gemma Qwen3.6

# # 9 从eval_results.jsonl提取hard并沉淀
# #   --input 可指定多个文件取并集；--clean 清理 augment 更新后失效的历史条目并重建 hard_{view}.json
# #   --reset-counts 清零所有 error_count（在重新跑 step 8 前执行）
# # python3 9_extract_errors.py --input eval_results.jsonl
# python3 9_extract_errors.py \
# --input \
# /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/tools/BAKUP/eval_results_v2_gemma.jsonl \
# /root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology/tools/BAKUP/eval_results_v2_qwen3.6.jsonl \
# --clean --reset-counts 

# 9.1 LLM 审核 Hard Negative 句子级有效性（--dry-run 只看不删）
# python3 9_1_clean_hard.py --host $HOST --port $PORT -w $WORKERS --dry-run --verbose --limit 4
python3 9_1_clean_hard.py --host $HOST --port $PORT -w $WORKERS
