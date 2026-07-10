"""远端切片/视频 VLM 审核的共享引擎 (stage2 视频 / stage3 切片复用)。

统一「远端拉取 → medoid 抽帧 → judge_frame 判定 → 拒绝则远端删」的双缓冲流水线:
拉取批 N+1 与审核批 N 并行 (审核吃 VLM/GPU, 拉取吃网络/IO, 互不阻塞)。

调用方只需提供:
  - RemoteAudit(远端配置) 实例;
  - next_files():  返回下一批待审文件名 (空列表 = 暂时无, 交由 poll 决定停/等);
  - on_results(res): 处理一批结果 dict{name: passed_bool} (记进度 / 远端删 / 黑名单 等)。

领域差异 (stage2 整段 vs stage3 切片) 落在调用方的 next_files/on_results, 引擎不感知。
"""
import os
import base64
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from llm_client import call_vlm_raw, frames_to_img_bytes
from representative_frame import triptych_reps_from_video
from lib import config
from lib import duration_filter
from lib.vlm_prompts import judge_frame

SSH_OPTS = config.SSH_OPTS   # 复用 config 统一定义 (2_3/3_1/remote_audit 一致)


class EndpointRouter:
    """least-inflight VLM 端点路由 (线程安全): 每次取当前在途最少的端点, 均衡负载。"""
    def __init__(self, eps):
        self.eps = eps
        self._inflight = [0] * len(eps)
        self._lock = threading.Lock()

    def pick(self) -> int:
        with self._lock:
            i = self._inflight.index(min(self._inflight))
            self._inflight[i] += 1
        return i

    def release(self, i: int):
        with self._lock:
            self._inflight[i] = max(0, self._inflight[i] - 1)


class RemoteAudit:
    """远端审核引擎: 绑定一个远端目录, 提供 SSH/枚举/删除/拉取/审核/流水线。"""

    def __init__(self, remote_host: str, remote_dir: str, shm_base: str, router: EndpointRouter):
        self.remote = remote_host
        self.remote_dir = remote_dir
        self.shm_base = shm_base
        self.router = router

    # ── 远端操作 ──
    def _ssh(self, script: str, timeout=30) -> subprocess.CompletedProcess:
        return subprocess.run(f"sshpass -e ssh {SSH_OPTS} {self.remote} bash",
                              shell=True, input=script, capture_output=True, text=True,
                              env=os.environ.copy(), timeout=timeout)

    def enumerate_remote(self, timeout=600) -> list[str]:
        """单次低开销枚举远端 .mp4 (ls -1U: 不排序/不 stat)。"""
        r = subprocess.run(f"sshpass -e ssh {SSH_OPTS} {self.remote} 'ls -1U {self.remote_dir}'",
                           shell=True, capture_output=True, text=True,
                           env=os.environ.copy(), timeout=timeout)
        return [l.strip() for l in r.stdout.splitlines() if l.strip().endswith(".mp4")]

    def remote_delete(self, names: list[str]):
        """批量删远端文件 (500/批, ./ 前缀防 dash 开头被当选项)。"""
        for i in range(0, len(names), 500):
            chunk = names[i:i + 500]
            self._ssh(f"cd '{self.remote_dir}' && rm -f -- " + " ".join(f"'./{f}'" for f in chunk))

    # ── 拉取 (线程池 + 指数退避重试, 收口"边删边新增"的瞬时失败) ──
    def _pull_one(self, name: str, shm: str, retries=5) -> bool:
        cmd = (f"sshpass -e rsync -aW --inplace --timeout=30 "
               f"-e 'ssh {SSH_OPTS}' '{self.remote}:{self.remote_dir}/{name}' '{shm}/{name}'")
        p = f"{shm}/{name}"
        for attempt in range(retries):
            try:
                subprocess.run(cmd, shell=True, capture_output=True, env=os.environ.copy(), timeout=60)
            except (subprocess.TimeoutExpired, OSError):
                pass
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return True
            if attempt < retries - 1:
                time.sleep(attempt + 1)
        return False

    def pull_batch(self, files: list[str], shm: str, workers=24) -> list[str]:
        if not files:
            return []
        os.makedirs(shm, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda n: self._pull_one(n, shm), files))
        return [f for f in os.listdir(shm) if f.endswith(".mp4")]

    # ── 单文件审核: 时长预闸 → 3段medoid多图 → judge_frame ──
    def audit_one(self, path: str) -> bool:
        if duration_filter.is_too_long(path) or duration_filter.is_too_short(path):
            return False   # 超长/过短直接判否 → 调用方远端删
        reps = triptych_reps_from_video(path, n_seg=3, fps=1.0, max_side=480)  # 头/中/尾各 medoid
        if not reps:
            return False
        b64s = []
        for fr in reps:
            ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                b64s.append(base64.b64encode(buf).decode())
        if not b64s:
            return False
        img_b = frames_to_img_bytes(b64s)   # 多图按序 (与喂视频帧同法), 非拼接
        i = self.router.pick()
        try:
            return judge_frame(self.router.eps[i], img_b)
        except Exception:
            return True    # VLM 异常保守保留
        finally:
            self.router.release(i)

    def _audit_batch(self, names: list[str], shm: str, concurrency: int) -> dict:
        res = {}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(self.audit_one, os.path.join(shm, f)): f for f in names}
            for fut in as_completed(futs):
                res[futs[fut]] = fut.result()
        return res

    # ── 双缓冲流水线 (拉 N+1 ∥ 审 N) ──
    def pipeline(self, next_files, on_results, concurrency: int,
                 pull_workers=24, poll=60):
        """next_files() -> list[str] 下一批待审 (空=暂无); on_results(dict{name:bool}) 处理结果。
        poll>0 常驻: 无新文件时轮询等待, 持续吃上游新产出 (与 2_3_sync/3_1_split 同构);
        poll=0 耗尽即停 (供 2_2 外层自管 recheck)。返回 (total_pass, total_reject)。"""
        shm_a, shm_b = f"{self.shm_base}_A", f"{self.shm_base}_B"
        total_pass = total_reject = batch = 0
        t0 = time.time()

        def pull_until(shm):
            """拉一批; poll>0 时空则轮询等待直到非空; poll=0 空则返回 []。"""
            got = self.pull_batch(next_files(), shm, pull_workers)
            while not got and poll:
                print(f"[info] 无新文件, {poll}s 后重试...", flush=True)
                time.sleep(poll)
                got = self.pull_batch(next_files(), shm, pull_workers)
            return got

        pulled = pull_until(shm_a)
        shm_curr, shm_next = shm_a, shm_b
        try:
            while pulled:
                batch += 1
                _p, _sc, _sn = list(pulled), shm_curr, shm_next
                files_next = next_files()
                res, pull_res = {}, []

                def do_audit(_pp=_p, _ss=_sc):
                    res.update(self._audit_batch(_pp, _ss, concurrency))
                    on_results(res)   # 记进度 + 远端删 + 领域动作 (由调用方实现)

                def do_pull(_fn=files_next, _ss=_sn):
                    nonlocal pull_res
                    pull_res = self.pull_batch(_fn, _ss, pull_workers) if _fn else []

                tb = time.time()
                ta = threading.Thread(target=do_audit); tp = threading.Thread(target=do_pull)
                ta.start(); tp.start(); ta.join(); tp.join()

                passed = sum(1 for ok in res.values() if ok)
                total_pass += passed; total_reject += len(res) - passed
                shutil.rmtree(_sc, ignore_errors=True)
                tot = total_pass + total_reject
                rate = tot / max(time.time() - t0, 1)
                print(f"[batch {batch}] {len(res)} clips | pass={passed} reject={len(res)-passed} "
                      f"| {time.time()-tb:.0f}s | 累计 {tot} ({total_pass/max(tot,1)*100:.0f}%通过) "
                      f"| {rate:.1f} clips/s", flush=True)

                shm_curr, shm_next = _sn, _sc
                pulled = pull_res
                if not pulled:                # 预拉的下一批为空: poll>0 常驻轮询, poll=0 收工
                    pulled = pull_until(shm_curr)
        finally:
            shutil.rmtree(shm_a, ignore_errors=True)
            shutil.rmtree(shm_b, ignore_errors=True)
        return total_pass, total_reject
