#!/usr/bin/env python3
"""scene-split 修复测试 (plain assert; dino python 跑: 有 cv2 + 系统 ffmpeg).
运行: /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/python \
        sport_ontology/videos/tests/test_scene_split_fix.py
"""
import os, sys, subprocess, tempfile, shutil, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "..", "3_1_scene_split.py")
FFBN_PATH = "/root/paddlejob/workspace/env_run/penghaotian/llm_infer/llm_train/muscle_wiki/lib/fetch_frames_by_name.py"


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


def test_select_push_names_exact_count_ok():
    m = load_mod()
    survivors = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11]   # Bq: 删了 7,9
    names, err = m.select_push_names("Bq", survivors, n_produced=12, n_original=12)
    assert err is None, f"段数精确相等应通过, got {err}"
    assert names == [f"Bq_{i}.mp4" for i in survivors], "只推幸存段名"
    assert "Bq_7.mp4" not in names and "Bq_9.mp4" not in names, "被删段不复活"


def test_select_push_names_count_mismatch_aborts():
    m = load_mod()
    # 重切 11 段但原始 12 => 段数不等 => 错位风险 => 中止 (即便 survivors 子集成立)
    names, err = m.select_push_names("X", [0, 1, 10], n_produced=11, n_original=12)
    assert names == [] and err is not None and "11" in err and "12" in err, \
        f"段数不等必须中止, got names={names} err={err}"


def test_select_push_names_subset_trap_aborts():
    m = load_mod()
    # 子集陷阱: 原13 重切12, max(survivor)=11<12 子集会误过; 但段数 12!=13 必须中止
    names, err = m.select_push_names("Y", [0, 5, 11], n_produced=12, n_original=13)
    assert names == [] and err is not None, "子集成立但段数不等 => 仍中止 (防中间合并错位)"


def test_select_push_names_empty_survivors():
    m = load_mod()
    names, err = m.select_push_names("Z", [], n_produced=5, n_original=5)
    assert names == [] and err is None, "无幸存段则跳过, 不算错"


def test_survivors_and_original_maps():
    m = load_mod()
    d = tempfile.mkdtemp()
    try:
        sq = os.path.join(d, "split_queue.txt")
        rl = os.path.join(d, "remote_split_list.txt")
        open(sq, "w").write("Bq_0.mp4\nBq_1.mp4\nBq_2.mp4\nOther_0.mp4\n")
        open(rl, "w").write("Bq_0.mp4\nBq_2.mp4\nOther_0.mp4\n")
        nmap = m.n_original_map(sq)
        smap = m.survivors_map(rl)
        assert nmap["Bq"] == 3, f"原始段数, got {nmap.get('Bq')}"
        assert smap["Bq"] == [0, 2], f"幸存索引排序, got {smap.get('Bq')}"
        assert nmap["Other"] == 1 and smap["Other"] == [0]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_fetch_gallery_no_duration_label():
    ff = load(FFBN_PATH, "ffbn")
    d = tempfile.mkdtemp()
    try:
        # 把模块的 OUT_DIR 指到临时目录, 造一个切片目录含 2 帧 (无 mp4)
        ff.OUT_DIR = d
        clip = os.path.join(d, "clipX")
        os.makedirs(clip)
        open(os.path.join(clip, "000.jpg"), "wb").write(b"\xff\xd8\xff\xd9")
        open(os.path.join(clip, "001.jpg"), "wb").write(b"\xff\xd8\xff\xd9")
        out_html = os.path.join(d, "index.html")
        ff.write_html(out_html)           # 1-arg, 扫描 OUT_DIR
        body = open(out_html, encoding="utf-8").read()
        assert "clipX" in body, "clip 名应在 gallery"
        assert "414" not in body, "不应出现任何时长数字 (该列已删)"
        assert "帧" in body, "应显示帧数 (如 '2 帧')"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_detect_segments_none_for_no_cut():
    """no_cut 视频 (单色无切点) detect_segments 必须返回 None, 这样 --replace 才能 SKIP
    (整片无关键帧吸附头部粘连, 无需重编码)。"""
    m = load_mod()
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, "solid.mp4")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-f", "lavfi",
             "-i", "color=c=green:s=320x240:d=4:r=30",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", src],
            capture_output=True, timeout=60)
        import cv2
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = nf / fps if fps > 0 else 0
        cap.release()
        assert m.detect_segments(src, fps, dur) is None, "单色无切点应返回 None (no_cut)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_replace_purge_decision_uses_480():
    """replace 侧的 PURGE 判定走 duration_filter.should_purge (阈值 480)。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib", "duration_filter.py")
    df = load(p, "duration_filter")
    assert df.should_purge(4483.0) is True, "75min 必删"
    assert df.should_purge(300.0) is False, "5min 不删"


def test_is_terminal_status():
    m = load_mod()
    assert m._is_terminal_status("abc: OK 覆盖 3/3 段") is True
    assert m._is_terminal_status("00z: PURGED 超长492s 删原片+28切片") is True
    assert m._is_terminal_status("-0p: ABORT -0p: 段数不等 重切=6 原始=5; ...") is True
    assert m._is_terminal_status("--6: SKIP no_cut 无切点无需修复") is True
    assert m._is_terminal_status("xy: SKIP 无幸存段 (清单)") is True
    assert m._is_terminal_status("ab: SKIP 原片拉取失败/不存在 (重试 10 次仍失败)") is False
    assert m._is_terminal_status("cd: PUSH-FAIL(rc=255) 覆盖 9/10 段") is False
    assert m._is_terminal_status("ef: PURGE-FAIL 超长523s 删除未确认, 未记录待重试") is False
    assert m._is_terminal_status("gh: ENCODE-FAIL 全部 1 段编码失败, 未推") is False
    assert m._is_terminal_status("ij: ERROR Command '...' timed out") is False
    assert m._is_terminal_status("kl: PURGED(dry-run) 将删原片+2切片 (超长4486s)") is False
    assert m._is_terminal_status("mn: DRY-RUN 将覆盖 10/12 段 [...]") is False


def test_stem_of():
    m = load_mod()
    assert m._stem_of("-L1yIZqIZws: PURGED 超长4486s 删原片+2切片") == "-L1yIZqIZws"
    assert m._stem_of("Bq-D1sSJgKI: OK 覆盖 10/12 段") == "Bq-D1sSJgKI"
    assert m._stem_of("_4rdls-fHc8: SKIP no_cut 无切点无需修复") == "_4rdls-fHc8"


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
