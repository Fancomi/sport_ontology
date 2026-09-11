# 备份清单 (Backup Manifest)

生成时间: 2026-08-13
工作目录: `/root/paddlejob/workspace/env_run/penghaotian/llm_infer/sport_ontology`

## 0. 状态结论

- **代码已全部上传 GitHub**, 本地与远程完全同步, 工作树干净, 无待推送内容。
- 远程仓库: `github.com/Fancomi/sport_ontology` (branch `main`)
- 远程最新提交: `e50a132` fix(videos): pin yt-dlp>=2026.8.19 — android_vr client 403 修复
- 本地 HEAD == origin/main == `e50a132`
- 磁盘上 309 个文件全部被 git 追踪; **未追踪文件 0 个, 被 gitignore 忽略的文件 0 个** —— 没有游离在版本控制外的散落产物。

## 1. 需备份内容汇总

| 项目 | 内容 | 数量 | 大小 |
|---|---|---|---|
| Git 追踪文件 | 项目全部代码、配置、数据、产物 | 309 个 | **128.2 MB** |
| `.git/` 本地仓库历史 | 完整提交历史与对象 (GitHub 上已有, 本地另存一份可离线恢复) | - | 37 MB |
| **磁盘合计** | | | **≈ 165 MB** |

## 2. 追踪文件按目录明细

| 目录 | 文件数 | 大小 | 占比 | 内容 |
|---|---|---|---|---|
| `videos/` | 113 | 111.33 MB | 86.9% | 视频采集流水线源码 + 数据 (kinetics/k400/k600/k700 数据集 CSV, 审计交付清单) |
| `tools/` | 81 | 11.81 MB | 9.2% | 本体构建工具 + slot_ontology 中英文图谱 JSON/PNG |
| `sport_ontology/` | 13 | 3.54 MB | 2.8% | Obsidian 本体库 (含 claudian 插件 main.js 3.2MB) |
| `vllm_deploy/` | 23 | 0.77 MB | 0.6% | vLLM 部署脚本 + benchmark 报告 (PNG) |
| `docs/` | 18 | 0.37 MB | 0.3% | 文档 |
| `books/` | 12 | 0.20 MB | 0.2% | 图文配对提取脚本 (T1_auto_pair.py 等) |
| `prompt_lab/` | 37 | 0.10 MB | 0.1% | prompt 实验 |
| `gpu_guard/` | 5 | 0.02 MB | - | GPU 守护脚本 |
| 根目录 | 7 | 0.03 MB | - | CLAUDE.md / AGENTS.md / llm-wiki-cn.md / .gitignore / README.md / .vscode / .claude |

## 3. 大型数据 / 产物 (Top, >100KB)

| 大小 | 文件 |
|---|---|
| 29.86 MB | `videos/data/fitness/deliverables/3_audit_kept.txt` |
| 20.76 MB | `videos/data/fitness/seeds/datasets/kinetics/train.csv` |
| 20.76 MB | `videos/data/fitness/seeds/datasets/k700_train.csv` |
| 13.93 MB | `videos/data/fitness/seeds/datasets/k600_train.csv` |
| 10.04 MB | `videos/data/fitness/seeds/datasets/k400_train.csv` |
| 7.39 MB | `videos/data/fitness/deliverables/3_audit_deleted.txt` |
| 6.96 MB | `tools/slot_ontology_en.json` |
| 3.18 MB | `sport_ontology/.obsidian/plugins/claudian/main.js` |
| 2.35 MB | `tools/slot_ontology_cn.json` |
| 1.49 MB | `videos/data/fitness/seeds/datasets/kinetics/test.csv` |
| 1.49 MB | `videos/data/fitness/seeds/datasets/k700_test.csv` |
| 1.41 MB | `videos/data/fitness/seeds/datasets/kinetics/val.csv` |
| 1.41 MB | `videos/data/fitness/seeds/datasets/k700_val.csv` |
| 1.15 MB | `videos/data/fitness/seeds/datasets/k600_val.csv` |
| 0.87 MB | `tools/slot_vocab_en.png` |
| 0.77 MB | `videos/data/fitness/seeds/datasets/k400_val.csv` |
| 0.63 MB | `tools/slot_vocab_cn.png` |
| 0.15 MB | `vllm_deploy/benchmark/report/*.png` (3 张) |
| 0.15 MB | `videos/data/fitness/seeds/keywords.txt` |
| 0.15 MB | `tools/slot_vocab_en.json` |
| 0.11 MB | `sport_ontology/.obsidian/plugins/claudian/styles.css` |

## 4. 备份方式建议

- 代码与产物: 已通过 git 上传 GitHub (含全部 309 文件与完整历史), 这就是主备份。
- 本地离线副本: `git clone --mirror <repo>` 或直接打包 `.git/` (37MB) + `git archive HEAD` 落一份 tar。
- 视频原始二进制文件: 仓库内**不含** (videos/data 仅文本/CSV, 无 mp4 等), 如需备份原始视频请另行确认存放位置。

## 5. 安全提醒

- 远程 URL 中内嵌了 GitHub 个人访问令牌 (`ghp_...`), 明文存在于本地 git 配置。建议尽快在 GitHub 撤销并轮换该 token, 改用 `gh auth login` 或 SSH 方式。
