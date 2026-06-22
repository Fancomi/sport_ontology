#!/usr/bin/env python3
"""scene-split 修复测试 (plain assert; dino python 跑: 有 cv2 + 系统 ffmpeg).
运行: /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/python \
        sport_ontology/videos/tests/test_scene_split_fix.py
"""
import os, sys, subprocess, tempfile, shutil, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "..", "3_1_scene_split.py")
FFBN_PATH = "/root/paddlejob/workspace/env_run/penghaotian/llm_infer/llm_train/tools/fetch_frames_by_name.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_mod():
    return load(MODULE_PATH, "scene_split")


def test_build_cut_cmd_reencode_and_trim():
    m = load_mod()
    cmd = m.build_cut_cmd("src.mp4", "out.mp4", 2.0, 5.0, 30.0, end_is_cut=True)
    s = " ".join(cmd)
    assert cmd.index("-ss") < cmd.index("-i"), "-ss 必须在 -i 前 (前置 seek)"
    assert "libx264" in s and "copy" not in cmd, "必须 libx264 重编码, 不能 stream copy"
    ti = cmd.index("-t") + 1
    assert abs(float(cmd[ti]) - (3.0 - 1.0/30.0)) < 1e-3, f"end_is_cut 减 1/fps, got {cmd[ti]}"


def test_build_cut_cmd_last_segment_no_trim():
    m = load_mod()
    cmd = m.build_cut_cmd("src.mp4", "out.mp4", 2.0, 5.0, 30.0, end_is_cut=False)
    ti = cmd.index("-t") + 1
    assert abs(float(cmd[ti]) - 3.0) < 1e-3, f"末段不减帧, got {cmd[ti]}"


def test_build_cut_cmd_zero_fps_safe():
    m = load_mod()
    cmd = m.build_cut_cmd("src.mp4", "out.mp4", 2.0, 5.0, 0.0, end_is_cut=True)
    ti = cmd.index("-t") + 1
    assert abs(float(cmd[ti]) - 3.0) < 1e-3, "fps=0 不减帧"


def test_build_cut_cmd_low_fps_no_negative_dur():
    m = load_mod()
    # 病态: fps=2 (1/fps=0.5) 且段长仅 0.4s -> 减 1/fps 会变负; 应兜底不减, -t 仍 0.4
    cmd = m.build_cut_cmd("src.mp4", "out.mp4", 1.0, 1.4, 2.0, end_is_cut=True)
    ti = cmd.index("-t") + 1
    assert float(cmd[ti]) > 0, f"-t 不能为负/零, got {cmd[ti]}"
    assert abs(float(cmd[ti]) - 0.4) < 1e-3, f"减帧致负时应不减, got {cmd[ti]}"


def _make_synthetic(path):
    """红 2s + 蓝 2s, 强制 GOP=45@30fps => 关键帧 0,1.5,3.0; 真实切点在 2.0s (落关键帧之间)."""
    subprocess.run(
        ["ffmpeg", "-nostdin",
         "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2:r=30",
         "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2:r=30",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
         "-c:v", "libx264", "-g", "45", "-keyint_min", "45", "-sc_threshold", "0",
         "-pix_fmt", "yuv420p", "-y", path],
        capture_output=True, timeout=60)


def _internal_cuts(path, thr=0.3):
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", path, "-vf",
         f"select='gt(scene,{thr})',metadata=print:file=-", "-an", "-f", "null", "-"],
        capture_output=True, text=True, timeout=60)
    return sum(1 for ln in r.stdout.split("\n") if "pts_time" in ln)


def test_reencode_cut_has_no_internal_snap():
    m = load_mod()
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, "src.mp4"); _make_synthetic(src)
        out = os.path.join(tmp, "seg.mp4")
        subprocess.run(m.build_cut_cmd(src, out, 2.0, 4.0, 30.0, end_is_cut=False),
                       capture_output=True, timeout=60)
        assert os.path.getsize(out) > 0, "切片应非空"
        assert _internal_cuts(out) == 0, "重编码切片内部不应有 >=0.3 硬切 (吸附消除)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_copy_baseline_does_snap():
    """对照: 老 -c copy 切法确有内部切点 (证明测试有判别力)."""
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, "src.mp4"); _make_synthetic(src)
        out = os.path.join(tmp, "copy.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-ss", "2.0", "-i", src, "-t", "2.0",
                        "-c", "copy", "-avoid_negative_ts", "1", "-y", out],
                       capture_output=True, timeout=60)
        assert _internal_cuts(out) >= 1, "copy 切法应吸附出内部切点 (基线对照)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
