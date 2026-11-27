"""The ported Butterworth action filter must match the minor's client filter exactly (tag minor-final)."""

import importlib.util
from pathlib import Path

import numpy as np

from drtc.control.action_filter import ButterworthFilter

spec = importlib.util.spec_from_file_location("legacy_action_filter", Path(__file__).parent / "fixtures" / "legacy_action_filter.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def test_matches_the_old_filter():
    rng = np.random.default_rng(0)
    # a smooth trajectory with chunk-seam jumps and jitter, 6 joints at 30 Hz
    t = np.arange(600) / 30
    x = np.stack([40 * np.sin(0.5 * t + j) for j in range(6)], axis=1)
    x[::50] += rng.normal(0, 5, (len(x[::50]), 6))
    x += rng.normal(0, 0.5, x.shape)
    old = legacy.ButterworthFilter(cutoff=3.0, order=2, fps=30.0, gain=1.4, past_buffer_size=5)
    new = ButterworthFilter(cutoff=3.0, fps=30.0, order=2, gain=1.4)
    for a in x:
        np.testing.assert_allclose(new.apply(a), old.apply(legacy.FilterContext(action=a)), rtol=0, atol=1e-12)
