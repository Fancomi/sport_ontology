# source /root/paddlejob/workspace/env_run/penghaotian/envs/vllm19/bin/activate

# ── 统一配置 ──────────────────────────────────────────────────────────────────
# 手动覆盖示例：PORT="8001,8002" WORKERS=2 bash run.sh
source "$(dirname "$0")/../vllm_deploy/detect_ports.sh"
# ─────────────────────────────────────────────────────────────────────────────

# T1 自动图文配对：书籍 MD → pairs_*.json（与 pair_extractor.html 格式完全兼容）
# 输出文件保存在各书目录下，命名为 pairs_<书名>_<日期>.json
# 可用 pair_extractor.html「导入已有标注」按钮载入校对

# ── 单本书 ────────────────────────────────────────────────────────────────────

# 处理指定书籍目录（路径相对于 datas/，或填绝对路径）
# python T1_auto_pair.py --book datas/施瓦辛格健身全书 $VLM

# ── 全量处理 ──────────────────────────────────────────────────────────────────

# 处理 datas/ 下所有书籍（顺序执行，每本完成后立即保存）
# python T1_auto_pair.py --all $VLM

# ── 调试 / 快速验证 ───────────────────────────────────────────────────────────

# 如果 VLM 尚未启动，可先用此命令验证端口是否可达
# curl http://$HOST:${PORT%%,*}/v1/models

python T1_auto_pair.py --book datas/施瓦辛格健身全书 $VLM
