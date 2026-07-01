# videos/data/badminton/ 数据字典

`videos/` 流水线的**羽毛球领域**数据/进度/名单。用 `DOMAIN=badminton` 选择本领域；
流水线代码与健身共用一套引擎（`lib/` + 编号脚本），领域差异（关键词/prompt/路径/时长）
集中在 `lib/domains_badminton.py`。目录结构与用法同 `../fitness/README.md`。

## 领域约束（与健身的差异）

- **目标**：10 万量级羽毛球比赛视频，时长 10s–3h。
- **只留**：固定机位拍摄的真人羽毛球比赛（业余/专业均可）。
- **拒绝**：技术讲解/教学/分析、说话头/解说、白板 PPT、卡通动画、观众席、标题卡/花字。
  判定由 `title_blacklist`（采集侧粗筛）+ VLM prompt（`1_4`/`2_2`/`3_2` 精筛）双重把关。
- **时长**：`clean_max_duration=10800`（3h 上限），`purge_max_duration=10800`（放开超长删除，
  保留长比赛，交给 `3_1_scene_split` 按场景切成短段）。健身域分别是 600 / 480。
- **caption**：描述赛场击球信息（击球类型/正反手/单双打/站位落点/动作姿态），
  而非健身的器械/发力部位。

## 存储隔离

| 层 | 健身 | 羽毛球 |
|----|------|--------|
| 本地大盘 | `…/datas/videos/` | `…/datas/badminton_videos/` |
| 远程原片 | `ral@10.109.83.30:/root/back_2/…/videos` | `ral@10.109.83.30:/root/datasets_0/…/badminton_videos` |
| 远程切片 | `…/videos_split` | `…/badminton_videos_split` |
| 工作区 | `data/fitness/` | `data/badminton/` |
| 多机 peer | 3 机 :8555 | 先单机（`peer_urls=[]`，扩机时填） |

## seeds/（本领域已建）

| 文件 | 含义 |
|------|------|
| `keywords.txt` | 羽毛球比赛搜索词（通用/BWF 赛事/单双打/明星选手/业余对打/多语言）。 |
| `channels_seed.txt` | 上传完整比赛录像的官方与赛事 YouTube 频道种子。 |
| `datasets/` | 空——羽毛球无合适 Kinetics 细粒度标签，`kinetics_labels` 留空跳过该源。 |

`deliverables/` `pipeline_state/` `logs/` 空起步，随流水线推进生成，语义同健身域。

## 运行（每阶段均需 `DOMAIN=badminton`）

```bash
DOMAIN=badminton bash 1_collect_filter.sh all          # 采集+缩略图+VLM筛选
DOMAIN=badminton bash 2_download.sh                     # 单机下载 (多机: 2_download.sh N K)
DOMAIN=badminton SSHPASS='3dvision' bash 2_3_sync_videos.sh   # 传远端
DOMAIN=badminton SSHPASS='3dvision' bash 3_scene_split.sh     # 场景切割
DOMAIN=badminton SSHPASS='3dvision' python3 3_2_audit_splits.py --finalize  # 切片审核+定稿
# stage4 caption 不在本次目标范围内
```
