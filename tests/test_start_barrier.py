"""启动屏障验收：准备 / 确认 / 首个真实样本的集成测试（自动验收入口）。

模拟设备控制每端延迟与失败时点。期望值一律独立推导：
- 成员集：由注入故障规格推导应成功集合，再与执行证据（record.json、
  磁盘文件、解码内容）双向核对，不使用任何默认值遮错；
- 真实起录位置：first_sample.t_local - 节点真实时钟偏移（ground truth），
  不采用协调器自己的 skew 估计。

运行入口: python3 -m pytest tests/test_start_barrier.py -v
"""
import glob
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multirec.coordinator import Coordinator, DEFAULT_NODES
from multirec import verify

ALL_IDS = [c["id"] for c in DEFAULT_NODES]
NODES_CFG = {c["id"]: {"color": c.get("color"), "tone": c.get("tone"),
                       "true_offset": c["true_offset"]}
             for c in DEFAULT_NODES}
SCHED_SLACK = 0.08
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_SAMPLE = os.path.join(REPO_ROOT, "recordings", "543f4e7adbd0")


def make_kernel(tmp_path, latency=None):
    k = Coordinator(recordings_dir=str(tmp_path / "rec"))
    for cfg in DEFAULT_NODES:
        k.add_node(cfg)
    for nid, (lo, hi) in (latency or {}).items():
        for link in (k.links_up[nid], k.links_down[nid]):
            link.min_latency, link.max_latency = lo, hi
    return k


def evs(rec, nid, etype):
    return [e for e in rec["events"] if e["node"] == nid
            and e["type"] == etype]


def all_evs(rec, etype):
    return [e for e in rec["events"] if e["type"] == etype]


def media_files(rec):
    d = os.path.dirname(rec["record_path"])
    return sorted(glob.glob(os.path.join(d, "*.mp4"))
                  + glob.glob(os.path.join(d, "*.wav")))


def evidence_members(rec):
    """成员集只认执行证据：有首个真实样本事件且无错误。无默认值。"""
    return {nid for nid, nr in rec["nodes"].items()
            if not nr.get("error") and evs(rec, nid, "first_sample")}


def assert_real_starts(rec, members):
    """真实起录位置从独立参照（ground-truth 时钟）推导并核对。"""
    t_start = rec["t_start_coord"]
    assert t_start is not None
    for nid in members:
        fs = rec["nodes"][nid]["first_sample"]  # 缺失即 KeyError，不遮错
        err = fs["err"]
        indep = fs["t_local"] - rec["nodes"][nid]["true_offset"]
        assert abs(fs["t_coord"] - indep) <= err + 0.02, nid
        assert abs(fs["t_coord"] - t_start) <= err + SCHED_SLACK, nid


def verify_against_disk(rec):
    disk = json.load(open(rec["record_path"]))
    return verify.verify_session(
        disk, nodes_cfg=NODES_CFG,
        base_dir=os.path.dirname(rec["record_path"]))


# --------------------------- 反例：准备失败，屏障未达到 ---------------------------

def test_barrier_not_reached_never_declares_group_start(tmp_path):
    """任一端准备失败且策略为取消时，不得宣称整组开始。"""
    k = make_kernel(tmp_path)
    k.nodes["cam2"].fail_at = "prepare"
    rec = k.run_session(ALL_IDS, duration=1.0, policy="cancel")
    assert rec["result"] == "cancelled"
    assert rec["t_start_coord"] is None          # 未宣称整组开始
    assert all_evs(rec, "start_command_sent") == []
    assert all_evs(rec, "first_sample") == []
    assert media_files(rec) == []
    assert evidence_members(rec) == set()
    # 持久化的执行证据与内存记录一致
    disk = json.load(open(rec["record_path"]))
    assert disk["result"] == "cancelled"
    assert disk["t_start_coord"] is None
    assert evidence_members(disk) == set()


# --------------------- 反例：确认包全部丢失 → 限定失败 ---------------------

def test_dropped_start_acks_bounded_failure(tmp_path):
    """故意丢全部 start_ack：必须在有限重试后限定失败，而不是一直等待。"""
    k = make_kernel(tmp_path)
    k.links_down["mic1"].drop = lambda msg: msg.get("type") == "start_ack"
    rec = k.run_session(ALL_IDS, duration=1.0, policy="cancel")
    assert rec["result"] == "cancelled"
    assert rec["nodes"]["mic1"]["error"] == "start ack timeout"
    # 重试有界：恰好 1 + ack_retries 次开始命令、ack_retries 次重试
    assert len(evs(rec, "mic1", "start_command_sent")) == 1 + 2
    assert len(evs(rec, "mic1", "start_ack_lost_retry")) == 2
    # 限定失败：在共同起点前后不久即放弃，而不是等 mic1 的样本
    assert rec["finished_dt"] < rec["t_start_dt"] + 0.5
    assert evs(rec, "mic1", "first_sample") == []
    assert evidence_members(rec) == set()


# -------------- 正例：受控每端延迟下，首个真实样本即录制起点 --------------

def test_controlled_latency_first_sample_is_start_point(tmp_path):
    """每端延迟受控且不对称时，首个真实样本仍落在共同起点上，
    已写入的音视频有可核对身份（帧序号 / 脉冲 / 颜色 / 音调）。"""
    latency = {"cam1": (0.10, 0.20), "mic2": (0.05, 0.15)}
    k = make_kernel(tmp_path, latency=latency)
    rec = k.run_session(ALL_IDS, duration=1.2, policy="cancel")
    assert rec["result"] == "ok", rec["result"]
    members = evidence_members(rec)
    assert members == set(ALL_IDS)
    t_start = rec["t_start_coord"]
    for nid in ALL_IDS:
        sent = evs(rec, nid, "start_command_sent")[0]
        ack = evs(rec, nid, "start_ack")[0]
        first = evs(rec, nid, "first_sample")[0]
        assert sent["t"] < ack["t"] < first["t"]   # 三类事件因果有序
        assert ack["t"] < t_start                  # 确认在边界之前
    assert_real_starts(rec, members)
    files = {os.path.splitext(os.path.basename(f))[0]
             for f in media_files(rec)}
    assert files == members                        # 成员集 ↔ 磁盘文件
    v = verify_against_disk(rec)
    assert v["ok"], {n: o["errors"] for n, o in v["nodes"].items()
                     if o["errors"]}
    assert set(v["members"]) == set(ALL_IDS)


# ----------------- 验收入口：准备失败 + 确认丢失构造正反例 -----------------

@pytest.mark.parametrize("fail_node,drop_ack,expected_result,expected_members", [
    (None, None, "ok", {"cam1", "cam2", "mic1", "mic2"}),   # 正例
    ("cam2", "mic1", "partial", {"cam1", "mic2"}),          # 反例
])
def test_acceptance_members_and_real_starts(tmp_path, fail_node, drop_ack,
                                            expected_result, expected_members):
    k = make_kernel(tmp_path)
    if fail_node:
        k.nodes[fail_node].fail_at = "prepare"
    if drop_ack:
        k.links_down[drop_ack].drop = \
            lambda msg: msg.get("type") == "start_ack"
    rec = k.run_session(ALL_IDS, duration=1.0, policy="partial")
    assert rec["result"] == expected_result
    # 成员集：执行证据 == 独立推导的期望集合
    members = evidence_members(rec)
    assert members == expected_members
    # 真实起录位置：从独立参照推导
    assert_real_starts(rec, expected_members)
    # 与磁盘执行证据核对（实际解码运算）
    v = verify_against_disk(rec)
    assert set(v["members"]) == expected_members
    for nid in expected_members:
        assert v["nodes"][nid]["ok"], v["nodes"][nid]["errors"]
    # 受控错误必须暴露坏结果，而不是被吞掉
    if fail_node:
        assert rec["nodes"][fail_node]["error"]
        assert fail_node not in members
    if drop_ack:
        assert rec["nodes"][drop_ack]["error"] == "start ack timeout"
        assert drop_ack not in members
        # 磁盘上的游离文件不能使其成为成员
        assert drop_ack in v["stray_files"]


# --------------------- 旧样例回归：运行到底，证据可核对 ---------------------

def test_committed_regression_sample_verifies_to_the_end():
    """入库的多端录制旧样例必须运行到底：成员集与真实起录位置
    对应到执行证据（record.json + 磁盘文件 + 解码内容）。"""
    rec = json.load(open(os.path.join(REPO_SAMPLE, "record.json")))
    assert rec["result"] == "ok"
    members = {nid for nid, nr in rec["nodes"].items()
               if nr.get("first_sample") and not nr.get("error")}
    files = {os.path.splitext(f)[0] for f in os.listdir(REPO_SAMPLE)
             if f.endswith((".mp4", ".wav"))}
    assert members == files == set(ALL_IDS)
    t_start = rec["t_start_coord"]
    for nid in members:
        fs = rec["nodes"][nid]["first_sample"]
        indep = fs["t_local"] - rec["nodes"][nid]["true_offset"]
        assert abs(fs["t_coord"] - indep) <= fs["err"] + 0.02, nid
        assert abs(fs["t_coord"] - t_start) <= fs["err"] + SCHED_SLACK, nid
    v = verify.verify_session(rec, nodes_cfg=NODES_CFG, base_dir=REPO_SAMPLE)
    assert v["ok"], {n: o["errors"] for n, o in v["nodes"].items()
                     if o["errors"]}
    assert set(v["members"]) == set(ALL_IDS)
