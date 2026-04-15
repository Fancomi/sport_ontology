source /home/baidu/envs/mineru/bin/activate
# source /home/baidu/envs/swift/bin/activate
# # 1. translate_wiki
# python3 1_translate_wiki.py

# # 2. 基于Gemma4的增强
# python3 2_augment_wiki.py --reverse

# # 3. 槽位收集
# python3 3_collect_slots.py

# # 4. 节点信息收集
# python3 4_fetch_vocab_info.py

# 5. LLM
python3 5_enrich_with_llm.py