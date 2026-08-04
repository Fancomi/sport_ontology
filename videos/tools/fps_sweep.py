"""全量帧率复查 (只拉文件头, 不下整片)。

背景: 帧率预闸 (lib/fps_filter) 加得比阶段二审核晚, 只对之后新审的生效 —— 已通过的
2.1 万条里仍有低帧率素材。全量复查若按「下整片再读」要传 883G, 而 mp4/webm 的
moov/头部通常在文件开头, 实测 512KB 即可用 PyAV 读出 fps。故只拉每个文件的前 1MB。

产物: pipeline_state/2_fps_sweep.jsonl  {name, fps, status}
  status=ok 且 fps<=15  -> 待删 (确定性: 帧率是文件客观属性)
  status!=ok (读不出)   -> 不删, 记录待复查 (与 fps_filter 的保守口径一致)
可中断续跑: 已在产物里的 name 不再重复拉取。
"""
import io
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import config

# 可用 FPS_SWEEP_HEAD 覆盖。1MB 够绝大多数 (moov 在头部时); 一轮扫完后有约 19% 读不出
# —— 那些是 moov 在文件尾部 (yt-dlp 未做 faststart) 或 webm 容器头结构不同, 加大 head
# 再扫一遍可救回一部分, 仍读不出的按保守口径不删。
HEAD_BYTES = int(os.environ.get("FPS_SWEEP_HEAD", 1024 * 1024))
OUT = config.STATE_DIR / os.environ.get("FPS_SWEEP_OUT", "2_fps_sweep.jsonl")
LIST = os.environ.get("FPS_SWEEP_LIST", "/tmp/remote_list.txt")


def head_bytes(name):
    cmd = ("sshpass -e ssh %s %s 'head -c %d %s/%s'"
           % (config.SSH_OPTS, config.DOMAIN.remote_host, HEAD_BYTES,
              config.DOMAIN.remote_videos, name))
    p = subprocess.run(cmd, shell=True, capture_output=True,
                       env=os.environ.copy(), timeout=120)
    return p.stdout


def probe(name):
    import av
    try:
        data = head_bytes(name)
        if not data:
            return name, None, "empty"
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.video:
                return name, None, "no_video_stream"
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate
            return name, (float(rate) if rate else None), "ok"
    except Exception as exc:
        return name, None, type(exc).__name__


def main():
    names = [l.strip() for l in open(LIST, encoding="utf-8") if l.strip()]
    done = set()
    if OUT.exists():
        for line in open(OUT, encoding="utf-8"):
            try:
                done.add(json.loads(line)["name"])
            except Exception:
                pass
    todo = [n for n in names if n not in done]
    print("远端 %d | 已查 %d | 本次 %d" % (len(names), len(done), len(todo)), flush=True)

    lock = threading.Lock()
    n = 0
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=24) as ex:
        for fut in as_completed([ex.submit(probe, x) for x in todo]):
            name, fps, status = fut.result()
            with lock:
                out.write(json.dumps({"name": name, "fps": fps, "status": status}) + "\n")
                n += 1
                if n % 500 == 0:
                    out.flush()
                    rate = n / max(time.time() - t0, 0.01)
                    print("  [%d/%d] %.0fs  %.1f/s  ETA %.0fmin"
                          % (n, len(todo), time.time() - t0, rate,
                             (len(todo) - n) / rate / 60), flush=True)
    print("SWEEP_DONE %d" % n, flush=True)


if __name__ == "__main__":
    main()
