"""tests/bench/speed_report.py: parsing LeRobot 0.7's training log and nvidia-smi's utilization log."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("speed_report", Path(__file__).parent / "bench" / "speed_report.py")
speed_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speed_report)

LOG = """\
INFO 2026-11-19 02:20:00 ot_train.py:850 Start offline training on a fixed dataset, with effective batch size: 64
INFO 2026-11-19 02:20:30 ot_train.py:905 step:10 smpl:640 ep:2 epch:0.03 loss:1.2 grdn:3.1 lr:1.0e-05 data_s:0.500 prep_s:0.050 updt_s:2.000 step_s:2.550 smp/s:25 mem_gb:70.10
INFO 2026-11-19 02:20:40 ot_train.py:905 step:60 smpl:4K ep:10 epch:0.17 loss:0.9 grdn:2.0 lr:2.0e-05 data_s:0.010 prep_s:0.040 updt_s:0.900 step_s:0.950 smp/s:67 mem_gb:76.40
INFO 2026-11-19 02:20:50 ot_train.py:905 step:1K smpl:64K ep:200 epch:2.80 loss:0.5 grdn:1.0 lr:2.5e-05 data_s:0.030 prep_s:0.040 updt_s:0.900 step_s:0.970 smp/s:66 mem_gb:76.49
"""

GPU = """\
2026/11/19 02:20:05.123, 40
2026/11/19 02:20:35.123, 90
2026/11/19 02:20:45.123, 100
2026/11/19 02:21:30.000, 0
"""


def test_windows_after_warmup_are_averaged():
    windows = speed_report.parse_log(LOG)
    assert [w["step"] for w in windows] == [10, 60, 1000]
    res = speed_report.summarize(windows, warmup=50, gpu=speed_report.parse_gpu(GPU))
    assert res["windows"] == 2
    assert res["step_s"] == pytest.approx(0.96)
    assert res["data_s"] == pytest.approx(0.02)
    assert res["peak_mem_gb"] == 76.49
    assert res["data_share"] == pytest.approx(0.021, abs=1e-3)
    assert res["gpu_util_pct"] == 95.0  # samples from the end of the warmup window to the last window


def test_no_window_after_warmup_is_an_error():
    with pytest.raises(ValueError):
        speed_report.summarize(speed_report.parse_log(LOG), warmup=5000)
