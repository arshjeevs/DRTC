"""Noise floor: how much the SAME (old) setup's outputs differ between two machines, on the same inputs.

    python tests/parity/compare_refs.py <reference A .pt> <reference B .pt>
If old-vs-new differences (check.py) are within this, they are numerical noise of the GPU/kernels, not a change
in what the model computes.
"""

import sys

import torch


def main() -> None:
    a, b = (torch.load(p, weights_only=False) for p in sys.argv[1:3])
    print(f"A: {a['versions']} ({sys.argv[1]})\nB: {b['versions']} ({sys.argv[2]})")
    worst_c = worst_a = 0.0
    for sa, sb in zip(a["samples"], b["samples"]):
        assert sa["index"] == sb["index"], "different frames"
        same_in = all(torch.equal(sa["input"][k], sb["input"][k]) for k in sa["input"] if k != "task")
        same_noise = all(torch.equal(x[2], y[2]) for x, y in zip(sa["draws"], sb["draws"]))
        dc = (sa["chunk"] - sb["chunk"]).abs().max().item()
        da = (sa["actions"] - sb["actions"]).abs().max().item()
        worst_c, worst_a = max(worst_c, dc), max(worst_a, da)
        print(f"frame {sa['index']:>6}: inputs {'same' if same_in else 'DIFFERENT'}, noise "
              f"{'same' if same_noise else 'different'}, max |chunk diff| {dc:.2e}, max |action diff| {da:.3f}")
    print(f"worst: chunk {worst_c:.2e}, action {worst_a:.3f}")


if __name__ == "__main__":
    main()
