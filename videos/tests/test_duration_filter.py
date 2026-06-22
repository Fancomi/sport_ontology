#!/usr/bin/env python3
"""duration_filter 测试 (plain assert; dino python).
运行: /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/python \
        sport_ontology/videos/tests/test_duration_filter.py
"""
import os, sys, subprocess, tempfile, shutil, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.join(HERE, "..", "lib", "duration_filter.py")


def load():
    spec = importlib.util.spec_from_file_location("duration_filter", MOD)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _make(path, dur):
    subprocess.run(["ffmpeg", "-nostdin", "-f", "lavfi",
                    "-i", f"color=c=blue:s=160x120:d={dur}:r=10",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", path],
                   capture_output=True, timeout=60)


def test_threshold_is_480():
    assert load().MAX_DURATION_SEC == 480.0


def test_actual_duration_reads_real():
    m = load(); d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "v.mp4"); _make(p, 4)
        dur = m.actual_duration(p)
        assert dur is not None and abs(dur - 4.0) < 0.3, f"got {dur}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_actual_duration_none_on_bad():
    m = load(); d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "notvideo.mp4"); open(p, "w").write("garbage")
        assert m.actual_duration(p) is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_is_too_long_short_false():
    m = load(); d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "v.mp4"); _make(p, 3)
        assert m.is_too_long(p) is False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_is_too_long_unreadable_false():
    """读不出时长 -> 不误删 (保守保留)。"""
    m = load(); d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "bad.mp4"); open(p, "w").write("x")
        assert m.is_too_long(p) is False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_should_purge_pure():
    m = load()
    assert m.should_purge(5000.0) is True
    assert m.should_purge(479.0) is False
    assert m.should_purge(480.0) is False   # 边界: 等于不删
    assert m.should_purge(None) is False     # 读不出不删


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
    print(f"--- {len(fns)-failed}/{len(fns)} passed ---")
    sys.exit(1 if failed else 0)
