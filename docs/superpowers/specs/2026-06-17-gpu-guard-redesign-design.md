# GPU 占卡守卫重设计 — infer 常驻 + STOP/CONT 门控 + 迟滞双阈值

日期: 2026-06-17
文件: `tools/gpu_guard.sh`

## 1. 背景与问题

`gpu_guard.sh` 的目的是**占卡防平台回收**:平台会回收"GPU 利用率长期为 0"的机器(已发生过回收 util=0、显存高的机器),所以在空闲卡上跑 `infer.py` 顶起利用率。一旦有正经任务用卡,infer 应让位。

本项目当前的正经任务是 `videos/run_caption.py`:8 个 sglang 一卡一个、占满 8 张物理卡做 caption。它是**突发型负载**,单卡利用率在 13~50% 之间剧烈摆动。

旧 guard 有两个致命问题:

1. **flapping(反复横跳)**: 旧逻辑用单一阈值(util>30% kill / <30% start)。caption 负载恰好在 30% 上下摆,导致 infer 每轮被 kill→重启→再 kill,反复生灭。infer 进程创建/销毁、CUDA 上下文重建开销巨大,且重启的 infer 还没跑热就又被杀,白白空转,同时持续干扰 caption。

2. **检测判据不可靠**: 中途尝试过用 `nvidia-smi pmon` 按进程测利用率,但发现两个坑——pmon 的 GPU 索引与进程真实物理卡错位;且跨 namespace 的 infer 进程对 pmon 完全不可见。这套方案对本机的多 namespace 拓扑不成立。

## 2. 目标

- **通用、应用无关**: guard 只认识自己起的 worker(infer),不需认识任何外部程序。换 sglang/分割/关键点/encoder训练/SMPL渲染/EGL 等任意负载,零改动。
- **消除 flapping**: infer 状态稳定,不再反复生灭。
- **严防 util=0**: 任何卡任何时刻都有一方(infer 或外部任务)在顶利用率,杜绝平台回收触发条件。
- **caption 优先**: caption 在跑时让它独占算力全速跑;它自己的利用率足以防回收。

## 3. 核心设计

两点改动相对旧 SIGSTOP 差分版:

1. **kill/start → STOP/CONT 常驻**: infer 进程常驻不杀,只在运行(CONT)/挂起(STOP)间切换。SIGCONT 是毫秒级恢复,无进程创建、无显存重分配、无 CUDA 上下文重建。flapping 的代价从"反复重启"降到"几乎免费的状态翻转"。

2. **单阈值 → 迟滞双阈值(hysteresis)**: 进挂起和回运行用两条不同的线,中间留缓冲带,根除横跳。

### 通用原理:自暂停差分

guard 唯一能 100% 认识的进程是它自己起的 worker。判据只有一句话:**把自己的 worker 全部 SIGSTOP、排空几秒,剩下的利用率就是纯外部负载**。不管外部是什么算法、在不在同一 namespace、占多少显存,"我让开后这张卡还有没有人在算"这个信号永远准。

### 状态机(per-card)

每张卡有目标态 `RUN | SUSP`,infer 进程常驻:

```
RUN  : infer 运行, 顶 util(外部此刻闲)
SUSP : infer 挂起, 让位(外部此刻忙)
```

### 迟滞双阈值

```
纯外部 util > T_HIGH(10%) → 目标 SUSP(外部确实忙, 让位)
纯外部 util < T_LOW(5%)   → 目标 RUN (外部确实闲, 占卡)
T_LOW ~ T_HIGH 灰区        → 维持上轮目标(防抖)
```

一张卡外部稳定在 20% 时:在 RUN 不翻(20<10 不成立)、在 SUSP 也不翻(20>5 不成立)——不再横跳。只有外部真正跨过上下线才切换。

阈值取 10/5 的依据:实测 caption 跑时纯负载 13~36%(最低 13),始终 >T_HIGH;跑完则 →0,<T_LOW。于是 caption 期间 infer 全程挂起(caption 自己的 13~36% 撑住 util),caption 一停 infer 立即全卡 CONT 顶满。两段都不会 util=0。

## 4. 配置参数

| 参数 | 值 | 说明 |
|---|---|---|
| `ROUND` | 30s | 一轮周期,远快于平台小时级回收判定 |
| `DRAIN` | 3s | SIGSTOP 后排空,等在途 CUDA kernel 退出(实测足够) |
| `PROBES × PROBE_GAP` | 4 × 1s | 排空后多次采样取最大值,修突发负载单次采样漏判 |
| `T_HIGH` | 10% | 纯外部 util 超过 → 挂起 infer |
| `T_LOW` | 5% | 纯外部 util 低于 → 唤醒 infer |
| `DRY_RUN` | 0 | 1=只记录决策不持久化(测试用) |

## 5. 每轮流程

```
1. SIGSTOP 全部存活 worker, 露出纯外部负载
2. 排空 DRAIN 秒
3. PROBES 次采样, 逐卡取 util 最大值
4. 逐卡迟滞决策更新目标态(灰区维持上轮)
5. 按目标态落地:
     - worker 已死 → 重启(若目标 SUSP, 起后即 STOP)   ← 崩溃恢复
     - 目标 RUN  → CONT
     - 目标 SUSP → 保持 STOP
6. sleep ROUND, 下一轮
```

平台回收为小时级,每轮探测那 ~3s 的全机 util 下探不触发回收(已与运维确认),故无需错峰逐卡探测——保持"全停→测→决策"的最简形式。

## 6. 边界与鲁棒性

- **worker 匹配末尾锚定** `infer\.py $1\$`: 防卡 1 误匹配卡 10/11…
- **STOP 态进程也能被清理**: 杀 infer 前先 `pkill -CONT` 再 `kill`,避免挂起态进程杀不掉。
- **崩溃自愈**: worker 死了下一轮自动重启;若该卡目标 SUSP,起后立即 STOP,保持让位语义。
- **单 daemon**: 后台化仅靠 `_GPU_GUARD_DAEMON` 环境变量,重复手动执行会起多个 daemon。重启前务必先 `pkill -f gpu_guard.sh`。
- **setsid 起 worker**: worker 脱离 guard 成独立会话,避免残留子 shell 顶着脚本名污染 `pgrep`。

## 7. 验证结果(2026-06-17 部署)

- DRY_RUN 跑一轮:caption 在跑,8 卡 ext_max 28~48% 全判 SUSP,不触碰真进程 ✓
- 正式部署:8 个 infer 全部存活且状态 `Ts`(挂起),caption 独占 8 卡 ✓
- 连续两轮全 SUSP,无 flapping ✓
- **caption 吞吐 57.4 → 83.1 win/s(+45%)**,剩余 56h → 28.4h ✓
- 单 daemon 确认,worker setsid 独立会话 ✓

## 8. 未来切换其他任务

guard 与具体任务解耦。换成分割/关键点/训练/渲染等任意 GPU 负载:无需改 guard。它会照常每轮自暂停探测纯外部负载——外部忙则 infer 让位、外部闲则 infer 占卡。唯一前提:外部任务真正消耗 GPU 算力(util 体现),而非只占显存。

