source /home/baidu/envs/mineru/bin/activate
# source /home/baidu/envs/swift/bin/activate

# 视频描述 / 处理预热
# python video_frames.py --max-side 768
# python video_frames.py --max-side 336
# python video_frames.py --prebuild --max-side 768
# python video_frames.py --prebuild --max-side 336

# # 1. translate_wiki
# python3 1_translate_wiki.py

# # 2. 基于Gemma4的增强
# python3 2_augment_wiki.py --reverse

# # 3. 槽位收集
# python3 3_collect_slots.py

# # 4. 节点信息收集
# python3 4_fetch_vocab_info.py

# 5. LLM
# python3 5_enrich_with_llm.py # 增强图谱
python3 5_1_clean_ontology.py # 进一步删减

# # 6. 图谱构建[可视化]
# python3 6_build_wiki.py
