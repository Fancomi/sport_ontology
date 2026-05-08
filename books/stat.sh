
# ── 参数说明 ───────────────────────────────────────────────────────────────────
# --csv FILE       导出统计表，不指定时默认输出到 data/stats.csv
#                  使用标准 CSV 引号转义，字段含逗号时自动加引号，Excel/pandas 可直接读取
# --dup            终端打印重复书对，可与 --csv 同时使用
# --dup-csv FILE   导出重复书对，不指定时默认输出到 data/dup.csv，可与 --csv / --dup 同时使用
# --dup-threshold  max_overlap 阈值（默认 0.5）
#                  图像文件名即 SHA-256，交集 = 内容完全相同的图片数量
#                  主指标 max(overlap_a, overlap_b)：任意一侧超过阈值 → 大概率是同一本书
#                  Jaccard 也会计算，但只作参考列输出到 CSV
# --min-inter N    最少共享图片数（默认 3），避免仅 1-2 张偶发同图的误报
# --min / --max    只显示对数 ≥N / ≤N 的书（仅影响统计表，不影响重复检测）
#
# CSV 关系列（contains）说明：
#   A⊇B  → A 包含了 B 的大部分图片，B 更像是 A 的子集（保留 A，考虑删 B）
#   B⊇A  → B 包含了 A 的大部分图片，A 更像是 B 的子集（保留 B，考虑删 A）
#   A≈B  → 双向重叠，高度相似（扫描版 vs 精排版等，需人工判断保留哪本）
#
# 注意：命名差异（扫描版 vs 精排版）无法从名称自动识别，需人工根据 contains 列和
#       对数（pairs）大小决定保留哪个版本（一般保留对数更多的精排版）。
# ─────────────────────────────────────────────────────────────────────────────

# 只打印统计表，同时输出 data/stats.csv
# python stat_pairs.py /root/paddlejob/workspace/env_run/penghaotian/datas/book_md

# 统计表 + 重复检测同时输出（终端 + data/stats.csv + data/dup.csv）
# python stat_pairs.py /root/paddlejob/workspace/env_run/penghaotian/datas/book_md --dup

# 指定输出路径
python stat_pairs.py /root/paddlejob/workspace/env_run/penghaotian/datas/book_md \
  --csv data/stats.csv --dup --dup-csv data/dup.csv
