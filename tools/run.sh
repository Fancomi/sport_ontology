# source /home/baidu/envs/mineru/bin/activate
# source /home/baidu/envs/swift/bin/activate

# 视频描述 / 处理预热
# python video_frames.py --max-side 768
# python video_frames.py --max-side 336
# python video_frames.py --prebuild --max-side 768
# python video_frames.py --prebuild --max-side 336

# # 1. 翻译metadata到metadata_cn
# python3 1_translate_wiki.py

# 2. 基于Gemma4, 增强metadata_cn到augment_xxx.json, check会调用2.1
python3 2_augment_wiki.py --check

# 2.1 基于Gemma4 校验augment_xxx.json的合规性

# # 3. 槽位收集: 从augment_xxx.json获取
# python3 3_collect_slots.py

# # 4. 节点信息收集: 从Wordnet
# python3 4_fetch_vocab_info.py

# # 5. LLM 增强图谱
# python3 5_enrich_with_llm.py # 增强图谱

# # 5.1 基于 LLM 清理 slot_ontology.json 中不恰当的混淆关系。 
# python3 5_1_clean_ontology.py

# # 6. 图谱构建[可视化]
# python3 6_build_wiki.py

# # 7. 生成混淆样本
# python3 7_gen_confusable.py # 基于图谱随机提取 -> 按概率提取

# # 8. 评测混淆判断 产出json
# python3 8_eval_confusable.py

# # 8.1 分析混淆判断结果
# python3 8_1_analyze.py --compare \
# BAKUP/eval_results_v2_gemma.jsonl \
# BAKUP/eval_results_v2_qwen3.6.jsonl \
# --labels Gemma Qwen3.6

# # 8.2 基于手动的图谱删减
# python3 8_2_cleanup_pairs.py


# # 9 从eval_results.jsonl提取hard并沉淀
# python3 9_extract_errors.py