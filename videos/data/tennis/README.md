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

```bash
DOMAIN=tennis bash 1_collect_filter.sh all
DOMAIN=tennis bash 2_download.sh 3 0
DOMAIN=tennis bash 3_scene_split.sh
```

- **阶段一**：`1_collect_filter.sh` 高召回的采集/展开/缩略图筛选（list/expand/thumbnail filtering）。
- **阶段二**：`2_download.sh` 下载并审核完整视频（downloads and audits full videos）。
- **阶段三**：`3_scene_split.sh` 场景切割并审核切片（splits and audits segments）。

所有路径隔离在 `data/tennis/` 下，加上远端网球存储根
（`ral@10.109.83.30:/root/datasets_0/penghaotian/datas/yt-dlp-downloads/tennis_videos`）。

完整命令参考:

```bash
DOMAIN=tennis bash 1_collect_filter.sh all          # 采集+缩略图+VLM筛选
DOMAIN=tennis bash 2_download.sh                     # 单机下载 (多机: 2_download.sh N K)
DOMAIN=tennis SSHPASS='3dvision' bash 2_3_sync_videos.sh   # 传远端
DOMAIN=tennis SSHPASS='3dvision' bash 3_scene_split.sh     # 场景切割
DOMAIN=tennis SSHPASS='3dvision' python3 3_2_audit_splits.py --finalize  # 切片审核+定稿
# stage4 caption 不在本次目标范围内
```
