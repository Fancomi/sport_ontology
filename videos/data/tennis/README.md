# videos/data/tennis/ 数据字典

`videos/` 流水线的**网球领域**数据/进度/名单。用 `DOMAIN=tennis` 选择本领域；
流水线代码与健身/羽毛球共用一套引擎（`lib/` + 编号脚本），领域差异（关键词/prompt/路径/时长）
集中在 `lib/domains_tennis.py`，审核策略复用 `lib/domain_policies.build_court_match_policy`
（与羽毛球同构的 court-match 结构化审核）。目录结构与用法同 `../badminton/README.md`。

## 领域约束（与羽毛球的差异）

- **目标**：固定机位拍摄的真人网球比赛（业余/专业均可），场地为网球场（而非羽毛球场）。
- **只留**：真实球场对打、机位为端线正后方高位广角固定主机位。
- **拒绝**：技术讲解/教学/分析、说话头/解说、白板/PPT/战术图示板、卡通动画、观众席/颁奖/采访、
  标题卡/花字、纯集锦快剪。判定由 `title_blacklist`（采集侧粗筛）+ VLM 结构化审核策略
  `court-match-tennis-v1`（`1_4`/`2_2`/`3_2` 精筛）双重把关。
- **时长**：`clean_max_duration=10800`（3h 上限），`purge_max_duration=10800`（放开超长删除，
  保留长比赛，交给 `3_1_scene_split` 按场景切成短段）。同羽毛球。
- **caption**：描述赛场击球信息（正手/反手/发球/截击/高压球/挑高球、单打或双打、
  站位底线/中场/网前、是否上网截击），而非健身的器械/发力部位。

## 存储隔离

| 层 | 健身 | 羽毛球 | 网球 |
|----|------|--------|------|
| 本地大盘 | `…/datas/videos/` | `…/datas/badminton_videos/` | `…/datas/tennis_videos/` |
| 远程原片 | `ral@10.109.83.30:/root/back_2/…/videos` | `ral@10.109.83.30:/root/datasets_0/…/badminton_videos` | `ral@10.109.83.30:/root/datasets_0/…/tennis_videos` |
| 远程切片 | `…/videos_split` | `…/badminton_videos_split` | `…/tennis_videos_split` |
| 工作区 | `data/fitness/` | `data/badminton/` | `data/tennis/` |
| 多机 peer | 3 机 :8555 | 单机（`peer_urls=[]`） | 单机（`peer_urls=[]`） |

## seeds/（本领域已建）

| 文件 | 含义 |
|------|------|
| `keywords.txt` | 网球比赛搜索词（通用/ATP·WTA·ITF 赛事/四大满贯/单双打/场地变体/明星选手/业余对打/多语言）。 |
| `channels_seed.txt` | 上传完整比赛录像的官方与赛事 YouTube 频道种子（ATP/WTA/四大满贯/Tennis TV/国家协会等）。 |
| `datasets/` | 空——网球无合适 Kinetics 细粒度标签，`kinetics_labels` 留空跳过该源。 |

`deliverables/` `pipeline_state/` `logs/` 空起步，随流水线推进生成，语义同健身/羽毛球域。

**提醒**：种子关键词/频道只是高召回的入池信号，不是最终分类器——命中关键词或频道的视频仍可能是
讲解/教学/集锦，真正的留存判定统一交给 VLM 结构化审核（`court-match-tennis-v1` 策略）。

## 运行（每阶段均需 `DOMAIN=tennis`）

**重要：`2_download.sh`、`2_3_sync_videos.sh`、`2_2_audit_videos.py` 是三个各自常驻/循环的
长跑进程，必须在各自独立的终端/后台任务中并发运行，而不是当作一个顺序执行到底的脚本列表。
下载写入本地盘 → 同步搬到远端 → 审核吃远端已同步的视频；三者持续并行才能让阶段二产出被审核过、
非空的留存视频。阶段三同理：`3_1_scene_split` 常驻切片，`3_2_audit_splits.py`（不带
`--finalize` 的**正常审核调用**）必须单独常驻运行去审这些切片；只有在正常审核已经产出
`3_audit_kept.txt`（非空）之后，才运行一次性的 `3_2_audit_splits.py --finalize`
去收敛权威名单——`--finalize` 本身**不审核**（`不审核`，见 `--finalize` 的 CLI 帮助），
它只是把「远端实际存在」与「已审核保留」两份清单做交集写盘。**如果跳过正常审核直接
`--finalize`，`canonical_segments.list` 会是空的，且所有远端切片会被报告为「漏网」。**

### 全新数据根的最小烟雾验证流程（产出非空 canonical 名单）

以下步骤给出一次可验证的最小端到端跑法：每个「常驻」步骤单开一个终端/`nohup ... &`
后台任务并保持运行，不要等它结束再继续下一步。

```bash
# 终端/后台任务 A — 阶段一：一次性批处理 (list/expand/thumbnail filtering, 跑完自然退出)
DOMAIN=tennis bash 1_collect_filter.sh all

# 终端/后台任务 B — 阶段二·下载：常驻循环 (每次退出后自动 5 分钟重启, 需手动 Ctrl-C/kill 停止)
DOMAIN=tennis bash 2_download.sh

# 终端/后台任务 C — 阶段二·同步：常驻循环 (--loop, 持续把本地已下载视频 rsync 到远端)
DOMAIN=tennis SSHPASS='3dvision' bash 2_3_sync_videos.sh

# 终端/后台任务 D — 阶段二·审核：常驻循环 (--recheck 默认 600s, 持续吃 C 新同步上来的视频)
DOMAIN=tennis SSHPASS='3dvision' python3 2_2_audit_videos.py

# 等 D 至少跑完一轮 (日志出现 "[轮 1]" 且 已审 > 0) 后，再启动阶段三：
# 终端/后台任务 E — 阶段三·切割：常驻循环 (拉远端已审视频切场景, 推切片回远端)
DOMAIN=tennis SSHPASS='3dvision' bash 3_scene_split.sh

# 终端/后台任务 F — 阶段三·正常审核 (常驻; 不加 --finalize, 真正跑 VLM 判定并落 3_audit_kept.txt)
DOMAIN=tennis SSHPASS='3dvision' python3 3_2_audit_splits.py

# 确认 F 已产出非空 data/tennis/deliverables/3_audit_kept.txt 后，
# 单独执行一次性的 finalize (只做「远端∩kept」收敛，不审核，可随时安全重跑):
DOMAIN=tennis SSHPASS='3dvision' python3 3_2_audit_splits.py --finalize
# 验证: canonical_segments.list 非空、"漏网"计数应为 0 (非 0 说明 F 还没审完，需先补审再 finalize)
wc -l data/tennis/deliverables/3_canonical_segments.list
```

- **阶段一**：`1_collect_filter.sh` 高召回的采集/展开/缩略图筛选（list/expand/thumbnail filtering），
  一次性批处理，跑完即退出。
- **阶段二**：三个独立长跑进程并发——`2_download.sh`（下载，自动重启循环）、
  `2_3_sync_videos.sh`（同步到远端，`--loop` 常驻）、`2_2_audit_videos.py`（**必须启动**，
  审核完整视频并在远端删除未通过项，`--recheck` 常驻轮询新同步的视频）。
  三者缺一都会导致「下载了但没审」或「审了但没同步」。
- **阶段三**：`3_scene_split.sh` 场景切割（常驻），随后**必须先跑正常的
  `3_2_audit_splits.py`（不带 `--finalize`）审核切片**，产出 `3_audit_kept.txt`/
  `3_audit_deleted.txt`；最后单独执行一次 `3_2_audit_splits.py --finalize`
  收敛出 `3_canonical_segments.list`（`--finalize` 只做集合运算，不调用 VLM、不产生新的
  kept/deleted 记录）。

所有路径隔离在 `data/tennis/` 下，加上远端网球存储根
（`ral@10.109.83.30:/root/datasets_0/penghaotian/datas/yt-dlp-downloads/tennis_videos`）。

完整命令参考（阶段二/三标注了哪些是需要保持运行的常驻进程）:

```bash
DOMAIN=tennis bash 1_collect_filter.sh all                     # 采集+缩略图+VLM筛选 (一次性)
DOMAIN=tennis bash 2_download.sh                                # [常驻] 下载 (多机: 2_download.sh N K)
DOMAIN=tennis SSHPASS='3dvision' bash 2_3_sync_videos.sh        # [常驻] 传远端
DOMAIN=tennis SSHPASS='3dvision' python3 2_2_audit_videos.py    # [常驻] 整段视频审核 (与上两者并发)
DOMAIN=tennis SSHPASS='3dvision' bash 3_scene_split.sh          # [常驻] 场景切割
DOMAIN=tennis SSHPASS='3dvision' python3 3_2_audit_splits.py    # [常驻] 切片正常审核 (先于 finalize)
DOMAIN=tennis SSHPASS='3dvision' python3 3_2_audit_splits.py --finalize  # [一次性] 收敛权威名单 (不审核)
# stage4 caption 不在本次目标范围内
```
