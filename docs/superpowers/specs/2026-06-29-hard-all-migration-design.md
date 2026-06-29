# hard_all 迁移 + 重评 设计

日期: 2026-06-29
状态: 已确认，实施中

## 背景

槽位体系 14→13（删 `limb_state` 折进 `posture_alignment`，新增 `body_position`/`tempo`），
标注员对上千条数据做了修正。新 aug 已从 `http://10.52.104.78:8555/datas/muscle_wiki/`
全量同步到本地（19532 个 JSON，含 reslot 后的 cat3）。

`hard_all` 负样本库（当前 + BAKUP）是基于旧 aug 文本构建的。数据本身未动，但因新 aug：
1. 部分 hard 失效（`original_value` 在新 aug 已不存在 → negative 的语义前提没了）；
2. 难度可能变化（需重评确认）。

## 失效量化（key_valid 真实判据）

| 文件 | 总条目 | 失效 | 失效率 |
|---|---|---|---|
| hard_all_cn | 57141 | 2125 | 3.7% |
| hard_all_en | 54431 | 14129 | 26.0% |

失效几乎全是 `value_changed`（标注真改了 ground truth，如 `双手`→`单手`、`hands`→`both hands`），
非措辞漂移。已确认：直接按 `key_valid` 删，不做 synonym 挽回（挽回会让 original_value 与新 aug
文本错位，后续 cloze 重评题干/选项错位，留错）。

## 方案

迁移逻辑极简，复用 `hard_utils` 现成 API（`load_hard_all` / `save_hard_all` / `clean_stale` /
`key_valid`），不造臃肿脚本。

### 组件 1：`tools/migrate_hard.py`（新增，一次性可复用迁移工具）

对 4 个文件逐一处理：当前 `hard_all_{cn,en}.jsonl` + `BAKUP/hard_all_{cn,en}_merged.jsonl`。

每文件三步：
1. **备份**：仅 BAKUP 两文件各拷 `*_premigrate_<YYYYMMDD>.jsonl`（当前 hard_all 别处已备份，不拷）。
2. **剔失效**：`clean_stale` 删 `[slot:orig]` 已不在新 aug 的条目。
3. **清零累计**：存活条目重置 `pred_count=0, error_count=0, pred_by_model={}, error_by_model={}`，
   `is_correct` 置 `None`（待重评）。其余字段（video/view/source/replaced_slot/original_value/
   new_value）保持不变。

参数：`--dry-run`（只报告不写盘）、`--lang`（限定语言，默认两种都跑）。
输出报告：每文件 删除/存活/失效率 + 按 slot 失效分布。

BAKUP 文件的 lang 通过文件名 `_cn_/_en_` 推断，`key_valid` 用对应 lang。

### 组件 2：重评（复用 `rescore_xmodel.sh`，不改）

迁移产出计数归零的干净 hard_all 后，运行 `rescore_xmodel.sh` 全量双模型 cloze 重打分：
`loop_cloze.sh` → `8_3_cloze_eval.py`（flush_hard_all 从 0 累加 pred/error_by_model）→
`9_extract_errors.py` 聚合。重评不需要新代码。

**重评独立于迁移**：迁移完先停、出报告，等用户确认 GPU/端口就绪再启重评。

## 能力边界

- `migrate_hard.py`：纯数据迁移（剔失效 + 清零），消费新 aug，产出干净 hard_all。
- `rescore_xmodel.sh`：重评，消费干净 hard_all。
- 两者解耦；migrate 可复用于未来任何 aug 大改后的 hard 迁移。

## 验证

1. `--dry-run` 报告，确认删除数 ≈ cn 2125 / en 14129，符合 3.7% / 26%。
2. 实跑后 `wc -l` 校验存活行数；抽查存活条目计数已清零。
3. BAKUP premigrate 备份存在。
