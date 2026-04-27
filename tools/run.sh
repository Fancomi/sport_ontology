# source /home/baidu/envs/mineru/bin/activate
# source /home/baidu/envs/swift/bin/activate

# ── 统一配置 ──────────────────────────────────────────────────────────────────
HOST="127.0.0.1"
PORT="8001,8002,8003,8004,8005,8006,8007,8008"
WORKERS=8
LANG="cn"                              # ← cn / en 切换语言
# ─────────────────────────────────────────────────────────────────────────────

# 视频描述 / 处理预热
# python video_frames.py --max-side 768
# python video_frames.py --prebuild --max-side 768

# # 1. 翻译metadata到metadata_cn
# python3 1_translate_wiki.py --host $HOST --port $PORT -w $WORKERS

# # 2. VLM 扩写视频描述（--check 启用 LLM 质检，质检复用同一组端口）
# python3 2_augment_wiki.py --host $HOST --port $PORT --check -w $WORKERS

# 2.1 批量校验 augment_xxx_cn.json 合规性 [已合入2混合调用]
# python3 2_1_check_augment.py --host $HOST --port $PORT -w $WORKERS

# # 2.2 将 augment_*_cn.json 中文描述翻译为英文，输出 augment_{view}_en.json
# #     --check  翻译后启动 LLM 语义 QC 自校正循环（最多 12 轮）
# #     _validated 仅在 QC 通过时写入，失败文件下次运行自动重试（无需 --reset-qc）
# python3 2_2_translate_augment.py --host $HOST --port $PORT -w $WORKERS --check --dry-run --limit 2
# for i in $(seq 1 5); do
#     python3 2_2_translate_augment.py --host $HOST --port $PORT -w $WORKERS --check
# done


# # # 3. 槽位统计: 覆盖输出 slot_vocab_{lang}.json / slot_vocab_{lang}.png
# python3 3_collect_slots.py --lang $LANG #[--top 20]

# # # 5. LLM 补充本体属性：读取 slot_vocab_{lang}.json，产出/更新 slot_ontology_{lang}.json
# # #    清理过期键、补充新词、保留已有条目
# python3 5_enrich_with_llm.py --host $HOST --port $PORT -w $WORKERS --lang $LANG

# # # 5.1 基于 LLM 清理 slot_ontology_{lang}.json 中不恰当的混淆关系。
# python3 5_1_clean_ontology.py --host $HOST --port $PORT -w $WORKERS --lang $LANG

# # # # 5.2 关系对称传播增强（无 LLM，纯集合运算，可反复运行直到 5_1 收敛）
# # # python3 5_2_infer_relations.py --lang $LANG  # 原地覆盖
# python3 5_2_infer_relations.py --lang $LANG
# # # 5.2 完成后建议重跑 5.1 清理传播引入的噪声（循环直到两脚本均无变化）
# for i in $(seq 1 5); do
#     python3 5_1_clean_ontology.py --host $HOST --port $PORT -w $WORKERS --lang $LANG
# done

# # 6. 图谱构建[可视化]
# python3 6_build_wiki.py --lang cn --force
# python3 6_build_wiki.py --lang en --force

# # 8. VLM 评测（挖掘 confusable / 评测 hard）
# #    --mode confusable  在线采样评测，结果追加 eval_results_{lang}.jsonl
# #    --mode hard        重刷累计 hard 分数（先跑 9 --reset-counts）
# #    --mode all         全部（默认）
# python3 8_eval_confusable.py --host $HOST --port $PORT -w $WORKERS --lang $LANG
# python3 8_eval_confusable.py --host $HOST --port $PORT -w $WORKERS --lang $LANG --mode confusable

# 评测 HARD, 读取hard_all_$LANG.jsonl
# python3 8_eval_confusable.py --host $HOST --port $PORT -w $WORKERS --lang $LANG --mode hard

# # 实验smoke
# python3 8_eval_confusable.py \
# --host $HOST --port $PORT -w $WORKERS \
# --lang $LANG --mode confusable --limit 40 \
# --out /tmp/timing_test.jsonl

# # 8.3 完形填空评测（在线抽样）
# #     --limit N  限制文件数（调试）；--dry-run 只看 prompt 不调 VLM
# python3 8_3_cloze_eval.py --host $HOST --port $PORT -w $WORKERS --lang $LANG --dry-run --limit 4
# python3 8_3_cloze_eval.py --host $HOST --port $PORT -w $WORKERS --lang $LANG --limit 4

# =======================================
# # 8.1 分析混淆判断结果（--out 和 --stats 可省略，自动在 input 同目录同 stem 生成）
# python3 8_1_analyze.py --input eval_results_hard_${LANG}.jsonl
# python3 8_1_analyze.py --input BAKUP/eval_results_hard_gemma源gemma测.jsonl
# python3 8_1_analyze.py --input BAKUP/eval_results_hard_gemma源Qwen36测.jsonl
# python3 8_1_analyze.py --input BAKUP/eval_results_hard_cn_Qwen36源gemma测.jsonl
# python3 8_1_analyze.py --input BAKUP/20260423_qwen3_6/
# python3 8_1_analyze.py --input BAKUP/20260422_gemma4/
python3 8_1_analyze.py --input BAKUP/20260425_qwen3_6_en/

# python3 8_1_analyze.py --compare \
# BAKUP/eval_results_v2_gemma.jsonl \
# BAKUP/eval_results_v2_qwen3.6.jsonl \
# --labels Gemma Qwen3.6

# =======================================
# # 9. 从 eval_results_{lang}.jsonl 提取答错对，幂等合入 hard_all_{lang}.jsonl
# #    --input  可指定多个文件取并集
# #    --clean  清理 augment 更新后 [slot:orig] 已失效的历史条目
# #    --reset-counts  清零所有 error_count（在重新跑 step 8 --mode hard 前执行）
# python3 9_extract_errors.py --lang $LANG --input eval_results_${LANG}.jsonl
# python3 9_extract_errors.py --lang $LANG --reset-counts
# python3 9_extract_errors.py --lang $LANG \
#     --input \
#     BAKUP/eval_results_v2_gemma.jsonl \
#     BAKUP/eval_results_v2_qwen3.6.jsonl \
#     --clean --reset-counts

# # 9.1 LLM 审核 hard_all_{lang}.jsonl 句子级有效性（删除上下文等价 / 视觉不可辨条目）
# #     --dry-run 只看判断结果，不写文件；--verbose 打印完整 prompt（配合 --limit）
# python3 9_1_clean_hard.py --host $HOST --port $PORT -w $WORKERS --lang $LANG --dry-run --verbose --limit 4
# python3 9_1_clean_hard.py --host $HOST --port $PORT -w $WORKERS --lang $LANG

# # 9.2 渲染 hard_all_{lang}.jsonl → hn_render_{view}.json（单向，供人工标注）
# #     输出到每个视频叶目录；hn_render 前缀表示渲染产物，区别于 hard_all 数据源
# #     兼容单槽替换型（confusable/incompatibility）和完形填空型（__cloze__）
# #     每条 pair 含 hard_key，标注完成后可按 key 写回 hard_all_{lang}.jsonl
# #
# #     渲染全量（覆盖已有文件）
# python3 9_2_render_hard.py --lang $LANG
# #     指定不同来源版本
# python3 9_2_render_hard.py --lang $LANG --input BAKUP/hard_all_v2.jsonl
# #     只渲染正面视角
# python3 9_2_render_hard.py --lang $LANG --views front
# #     删除所有渲染文件
# python3 9_2_render_hard.py --lang $LANG --clean
