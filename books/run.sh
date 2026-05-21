# source /root/paddlejob/workspace/env_run/penghaotian/envs/vllm19/bin/activate

# ── 统一配置 ──────────────────────────────────────────────────────────────────
THINK=0  # 0=关闭 thinking，1=开启（质量↑，速度↓）
source "$(dirname "$0")/../vllm_deploy/detect_ports.sh" # 这里会将THINK转化到THINK_FLAG
# # ─────────────────────────────────────────────────────────────────────────────

# # T1 自动图文配对：书籍 MD → pairs_*.json（与 pair_extractor.html 格式完全兼容）
# # 输出文件保存在各书目录下，命名为 pairs_<书名>_<日期>.json
# # 可用 pair_extractor.html「导入已有标注」按钮载入校对

# # ── 单本书 ────────────────────────────────────────────────────────────────────
# # python T1_auto_pair.py --book datas/施瓦辛格健身全书 $VLM $THINK_FLAG

# # ── 内置 datas/ 全量 ──────────────────────────────────────────────────────────
# # python T1_auto_pair.py --all $VLM $THINK_FLAG

# # ── 外部目录全量（669 本书等大批量场景）─────────────────────────────────────────
# # # shellcheck disable=SC2086
# # python T1_auto_pair.py --dir /root/paddlejob/workspace/env_run/penghaotian/datas/book_md $VLM $THINK_FLAG

# # 新版 T1 默认只处理人工保留的 264 本书，按「文本段抽取 → 图片 caption → 图片成组 → 图文匹配」四阶段运行：
# # shellcheck disable=SC2086
# python T1_auto_pair.py --review $VLM $THINK_FLAG -w 32
#
# # 小规模测试示例：只跑 1 本书、2 个文本窗口、20 张图，输出 pairs_test_*.json：
# # shellcheck disable=SC2086
# python T1_auto_pair.py --review $VLM $THINK_FLAG --limit-books 1 --limit-windows 2 --limit-images 20 --out-prefix pairs_test -w 4 --force

# # 转为数据集
# python build_dataset.py \
# --src /root/paddlejob/workspace/env_run/penghaotian/datas/book_md \
# --out /root/paddlejob/workspace/env_run/penghaotian/datas/book_20260507/annotations \
# --val-ratio 0.10 \
# --size 768 \
# -w 16

# # ====================================
# python T2_recaption.py --dir /root/paddlejob/workspace/env_run/penghaotian/datas/book_md $VLM # --force


# # 转为数据集
# python build_dataset.py \
# --src /root/paddlejob/workspace/env_run/penghaotian/datas/book_md \
# --out /root/paddlejob/workspace/env_run/penghaotian/datas/book_20260507/annotations \
# --source recaption \
# --val-ratio 0.10 \
# --size 768 \
# -w 16

python build_dataset.py \
--src /root/paddlejob/workspace/env_run/penghaotian/datas/book_md \
--out /root/paddlejob/workspace/env_run/penghaotian/datas/book_20260508_clean/annotations \
--allowlist data/book_review_remain_20260508_185423.csv \
--source recaption \
--val-ratio 0.10 \
--size 768 \
-w 16
