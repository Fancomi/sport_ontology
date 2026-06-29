import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import audit_stages as a


def test_gate_multiattr_pass():
    attrs = {"has_person": True, "is_exercising": True, "scene_type": "real_person"}
    assert a.gate_decision(attrs, "V1") is True
    assert a.gate_decision(attrs, "V2") is True
    assert a.gate_decision(attrs, "V3") is True


def test_gate_multiattr_no_person_rejects():
    attrs = {"has_person": False, "is_exercising": False, "scene_type": "landscape"}
    assert a.gate_decision(attrs, "V2") is False


def test_gate_multiattr_person_but_not_real_scene_rejects():
    attrs = {"has_person": True, "is_exercising": True, "scene_type": "animation"}
    assert a.gate_decision(attrs, "V2") is False


def test_gate_multiattr_person_idle_rejects():
    attrs = {"has_person": True, "is_exercising": False, "scene_type": "real_person"}
    assert a.gate_decision(attrs, "V3") is False


def test_gate_minimal_v4():
    assert a.gate_decision({"has_person": True, "is_exercising": True}, "V4") is True
    assert a.gate_decision({"has_person": False, "is_exercising": True}, "V4") is False
    assert a.gate_decision({"has_person": True, "is_exercising": False}, "V4") is False


def test_gate_missing_keys_rejects():
    assert a.gate_decision({}, "V2") is False
    assert a.gate_decision({"has_person": True}, "V4") is False
