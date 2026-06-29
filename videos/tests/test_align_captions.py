import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)
import align_captions as ac


def test_plan_alignment_basic():
    # canonical 有 a,b,c ; 磁盘有 a,b,d (d=孤儿, c=缺口)
    canonical = {"a_0", "b_1", "c_2"}
    disk = {"a_0", "b_1", "d_3"}
    plan = ac.plan_alignment(canonical, disk)
    assert plan["orphans"] == {"d_3"}, plan["orphans"]      # 磁盘有∖canonical无
    assert plan["gap"] == {"c_2"}, plan["gap"]              # canonical有∖磁盘无
    assert plan["aligned"] == {"a_0", "b_1"}, plan["aligned"]  # 交集 = 对齐后保留


def test_plan_alignment_strips_mp4():
    # 真实场景: canonical 带 .mp4, 磁盘 JSON stem 不带 -> 规范化后应对齐
    canonical = {"a_0.mp4", "b_1.mp4", "c_2.mp4"}   # 切片名单 (带 .mp4)
    disk = {"a_0", "b_1", "d_3"}                     # JSON 文件名去 .json (不带 .mp4)
    plan = ac.plan_alignment(canonical, disk)
    assert plan["aligned"] == {"a_0", "b_1"}, plan["aligned"]  # a,b 命中 (后缀已规范化)
    assert plan["orphans"] == {"d_3"}, plan["orphans"]         # d 磁盘有∖权威无
    assert plan["gap"] == {"c_2"}, plan["gap"]                 # c 权威有∖磁盘无 (无 .mp4)


def test_plan_alignment_identity_check():
    # aligned + gap == canonical ; aligned == disk - orphans
    canonical = {"x_0", "y_1", "z_2", "w_3"}
    disk = {"x_0", "y_1", "extra_9"}
    plan = ac.plan_alignment(canonical, disk)
    assert plan["aligned"] | plan["gap"] == canonical
    assert plan["aligned"] == disk - plan["orphans"]


def test_scan_disk_stems(tmp_path):
    # 构造 captions/{shard}/{stem}.json, 验证扫描取 stem
    cap = tmp_path / "captions"
    (cap / "00").mkdir(parents=True)
    (cap / "ab").mkdir(parents=True)
    (cap / "00" / "vid1_0.json").write_text("{}")
    (cap / "ab" / "vid2_5.json").write_text("{}")
    (cap / "ab" / "_orphan").mkdir()                 # _orphan 子目录应被跳过
    (cap / "ab" / "_orphan" / "old_9.json").write_text("{}")
    stems = ac.scan_disk_stems(str(cap))
    assert stems == {"vid1_0", "vid2_5"}, stems


def test_move_orphans(tmp_path):
    cap = tmp_path / "captions"
    (cap / "b1").mkdir(parents=True)
    src = cap / "b1" / "orph_1.json"
    src.write_text("{}")
    moved = ac.move_orphans(str(cap), {"orph_1"})
    assert moved == 1
    assert not src.exists()
    assert (cap / "_orphan" / "b1" / "orph_1.json").exists()
