# AGENTS.md — GPU 利用率检测 / 占卡守卫工具

本目录提供两样东西,任何需要"判断 GPU 卡忙不忙"或"占卡防回收"的方法都应复用,**不要再自己写 `nvidia-smi` 一次性采样**(会被突发负载骗)。

| 文件 | 作用 |
|---|---|
| `gpu_util.py` | **GPU 真实利用率检测接口**(Python import + CLI),供其他方法调用 |
| `gpu_guard.sh` | 占卡守卫:空闲卡常驻 `infer.py` 顶利用率,外部任务用卡时自动让位 |
| `infer.py` | guard 拉起的占位负载(吃满单卡算力的死循环) |

---

## 一、检测接口 `gpu_util.py`

### Python 调用(推荐其他方法用这个)

```python
import sys; sys.path.insert(0, ".../sport_ontology/gpu_guard")
from gpu_util import sample_max, busy_cards, is_busy, external_max

util = sample_max()                  # {0:47, 1:12, ...} 每卡 max util%
busy = busy_cards(thresh=10)         # {0,3,5} 超阈值的忙碌卡集合
if is_busy(3, thresh=10): ...        # 单卡是否忙
ext  = external_max([1234, 5678])    # 暂停自己这些PID后, 测纯外部负载
```

### CLI 调用

```bash
python gpu_util.py                        # 每卡 max util
python gpu_util.py --busy --thresh 10     # 只打印忙碌卡号: 0 1 4
python gpu_util.py --json --samples 8 --gap 0.5
python gpu_util.py --exclude 1234,5678    # 自暂停差分: 排除自有PID测纯外部
```

### 参数
- `samples`(默认 4)/`gap`(默认 1.0s):采样次数与间隔。**逐卡取最大值**。
- `thresh`(默认 10):忙碌阈值 %。
- `external_max(exclude_pids, drain=3.0)`:SIGSTOP 自有 PID → 排空 `drain` 秒 → 采样 → SIGCONT。

---

## 二、要求(务必遵守)

1. **判忙必须用 `sample_max`,不要单次 `nvidia-smi`。** 突发型负载(VLM/sglang caption/渲染/训练)利用率在 0~100% 剧烈摆动,单次采样十有八九踩到间隙读出假的低值。多次取 max 才能捕到真实峰值。

2. **想知道"卡对【我】空不空",用 `external_max` 传入自己的 PID。** 整卡 util 是所有进程混合值;只有暂停自己后测残余,才是"别人是否在用这张卡"。

3. **判空闲要看利用率,不能只看显存。** vLLM/sglang 这类常占满显存但 sm≈0(没真跑)。显存高 util=0 的机器**会被平台回收**。判据永远是利用率。

4. **阈值留迟滞,别用单阈值反复判。** 若据此做"起/停"决策,进入和退出用两条不同的线(如 >10% 算忙 / <5% 算闲),中间留灰区维持现状,否则负载在阈值上抖动会导致决策反复横跳(flapping)。

---

## 三、避坑指南(都是踩过的真坑)

- **坑1:`nvidia-smi pmon` 的 GPU 索引与进程真实物理卡错位。** 别用 pmon 的卡号做归属判断。要 PID↔物理卡映射,用 `--query-compute-apps=gpu_uuid,pid` + `--query-gpu=uuid,index` 经 UUID 对齐。

- **坑2:跨 namespace 进程对 `nvidia-smi`/`pmon` 不可见。** 别的容器里的进程不出现在进程列表里,却抬高整卡 util。所以"按进程列表算外部占用"在多 namespace 机器上不成立——**唯一可靠的纯外部利用率测法是自暂停差分**(`external_max`):把自己让开,剩下的就是外部,不管它在哪个 namespace。

- **坑3:被 SIGSTOP 挂起的进程杀不掉。** 清理可能处于挂起态(`stat=T`)的进程,先 `pkill -CONT` 再 `kill`,否则信号递不到。

- **坑4:`infer.py N` 的 N 是逻辑卡号,匹配进程要末尾锚定。** 用 `pgrep -f "infer\.py $N\$"`,否则 `infer.py 1` 会误匹配 `infer.py 10/11`。

---

## 四、占卡守卫 `gpu_guard.sh`

空闲卡占位防回收,**应用无关**(外部跑 sglang/分割/关键点/训练/SMPL/EGL 均适用,零改动)。

- **启动**:`bash gpu_guard.sh`(自动后台化,打印 PID 与日志路径)
- **停止**:`kill <PID>`
- **重启前务必先 `pkill -f gpu_guard.sh`**:防重入只靠环境变量,重复执行会起多个 daemon 互相打架。
- **测试**:`DRY_RUN=1 bash gpu_guard.sh` 只记录决策不真动进程。

机制:每轮(30s)SIGSTOP 全部 infer → 排空 → 多采样取 max(纯外部负载)→ 迟滞双阈值逐卡决策(>10% 让位保持挂起 / <5% 占卡 CONT / 灰区维持)。infer 常驻只 STOP/CONT 不 kill,故无 flapping。平台回收为小时级,每轮探测那几秒的 util 下探不触发回收。

> 设计详情见 `docs/superpowers/specs/2026-06-17-gpu-guard-redesign-design.md`。
