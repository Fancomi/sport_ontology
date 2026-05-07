
# # 全量统计（默认按对数降序）
# python stat_pairs.py /root/paddlejob/workspace/env_run/penghaotian/datas/book_md

# # 只看 ≥50 对的书
# python stat_pairs.py /root/paddlejob/workspace/env_run/penghaotian/datas/book_md --min 50

# # 只看 ≤5 对的（可能质量差或不相关）
# python stat_pairs.py /root/paddlejob/workspace/env_run/penghaotian/datas/book_md --max 5

# 导出 CSV 方便 Excel 筛选
python stat_pairs.py /root/paddlejob/workspace/env_run/penghaotian/datas/book_md --csv stats.csv
