"""Eval files (lab/evals/<eval>.yaml): one study's fixed plan (lab/EVALUATION.md).

An eval names the models, the conditions (experiment settings from lab/experiments), the trials per model and
condition (METRICS.md rule 1), the success rule and the time limit. `lab eval freeze` pins every model to the Hub
commit it will be served at; an official trial only runs against a frozen plan.

Order (METRICS.md rule 2): trials run in rounds. A round holds every model x condition pair that has the fewest valid
trials so far, in a random order seeded by (seed, round number), so every machine computes the same order and a
voided trial is simply run again in a later round.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from drtc.registry import Registry, RegistryError

FROZEN_HEADER = "# Written by `lab eval freeze`: the Hub commit each model is served at. Do not edit by hand."


class EvalSpec(BaseModel):
    """One eval file."""

    model_config = ConfigDict(extra="forbid")
    description: str
    task: str | None = None  # the instruction; default: the models' dataset task (they must agree)
    success: str  # the success rule, fixed before the first trial
    time_limit_s: float
    trials: int = 20
    seed: int = 0
    models: list[str]
    conditions: dict[str, str]  # condition name -> experiment (lab/experiments/<group>/<name>)
    video: bool = True
    frozen: dict[str, str] = Field(default_factory=dict)  # model -> Hub commit (lab eval freeze)

    def pairs(self) -> list[tuple[str, str]]:
        """Every (condition, model) of the plan, in file order."""
        return [(c, m) for c in self.conditions for m in self.models]


def evals_dir(reg: Registry) -> Path:
    return reg.lab / "evals"


def eval_path(reg: Registry, name: str) -> Path:
    path = evals_dir(reg) / f"{name}.yaml"
    if not path.is_file():
        have = sorted(p.stem for p in evals_dir(reg).glob("*.yaml")) if evals_dir(reg).is_dir() else []
        raise RegistryError(f"no eval {name!r} (lab/evals/{name}.yaml); evals: {', '.join(have) or 'none yet'}")
    return path


def load(reg: Registry, name: str) -> EvalSpec:
    path = eval_path(reg, name)
    try:
        return EvalSpec.model_validate(yaml.safe_load(path.read_text()) or {})
    except Exception as e:  # pydantic's message names the bad field
        raise RegistryError(f"lab/evals/{name}.yaml: {e}") from e


def check(reg: Registry, spec: EvalSpec) -> list[str]:
    """Problems with an eval against the registry (empty = fine)."""
    problems = []
    if spec.trials < 1:
        problems.append("trials must be at least 1")
    if spec.time_limit_s <= 0:
        problems.append("time_limit_s must be positive")
    if len(set(spec.models)) != len(spec.models):
        problems.append("a model is listed twice")
    if not spec.models or not spec.conditions:
        problems.append("needs at least one model and one condition")
    tasks = set()
    for m in spec.models:
        if m not in reg.models:
            problems.append(f"model {m!r} is not in lab/models")
            continue
        r = reg.resolve(m)
        if r.dataset is None:
            problems.append(f"model {m!r} has no dataset (its units and cameras are unknown)")
        elif r.dataset.task:
            tasks.add(r.dataset.task)
    if spec.task is None and len(tasks) > 1:
        problems.append(f"the models' datasets have different tasks ({sorted(tasks)}); set `task:` in the eval")
    if spec.task is None and not tasks:
        problems.append("no task: set `task:` in the eval")
    for c, exp in spec.conditions.items():
        if not re.fullmatch(r"[a-z0-9_]+", c):
            problems.append(f"condition name {c!r}: use lower case letters, digits and _")
        try:
            reg.experiment_path(exp)
        except RegistryError as e:
            problems.append(f"condition {c!r}: {e}")
    for m in spec.frozen:
        if m not in spec.models:
            problems.append(f"frozen lists {m!r}, which is not in models")
    return problems


def task_of(reg: Registry, spec: EvalSpec) -> str:
    if spec.task:
        return spec.task
    return reg.resolve(spec.models[0]).dataset.task


def is_frozen(spec: EvalSpec) -> bool:
    return bool(spec.models) and all(spec.frozen.get(m) for m in spec.models)


def hub_commits(reg: Registry, models: list[str]) -> dict[str, str]:
    """The current Hub commit of each model's repo (what `lab serve` would serve)."""
    from huggingface_hub import HfApi

    api = HfApi()
    out = {}
    for m in models:
        repo = reg.resolve(m).hub
        try:
            out[m] = api.model_info(repo).sha
        except Exception as e:  # noqa: BLE001 - say which model, not a stack trace
            raise RegistryError(f"{m}: cannot read {repo} on the Hub ({type(e).__name__}); is it trained and pushed?")
    return out


def write_frozen(path: Path, frozen: dict[str, str]) -> None:
    """Replace the file's `frozen:` block (and its header comment), keeping everything else as written."""
    text = path.read_text()
    text = re.sub(rf"(?m)^{re.escape(FROZEN_HEADER)}\n", "", text)
    text = re.sub(r"(?m)^frozen:.*\n(?:[ \t]+.*\n?)*", "", text).rstrip("\n") + "\n"
    block = "\n".join(f"  {m}: '{sha}'" for m, sha in frozen.items())  # quoted: a hash may be all digits
    path.write_text(f"{text}\n{FROZEN_HEADER}\nfrozen:\n{block}\n")


def next_round(spec: EvalSpec, valid: dict[tuple[str, str], int]) -> tuple[int, list[tuple[str, str]]]:
    """The next round: its number (1-based) and its (condition, model) pairs in their seeded order.

    Returns (0, []) when every pair has its trials. `valid` counts the valid trials per pair so far.
    """
    counts = {p: valid.get(p, 0) for p in spec.pairs()}
    low = min(counts.values())
    if low >= spec.trials:
        return 0, []
    number = low + 1
    pairs = [p for p in spec.pairs() if counts[p] == low]
    random.Random(f"{spec.seed}-{number}").shuffle(pairs)
    return number, pairs
