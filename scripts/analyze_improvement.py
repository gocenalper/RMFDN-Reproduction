#!/usr/bin/env python3
"""Tables and figure for the improvement experiment (report Section 7.2-7.3).

Reads, for each seed, a suite directory written by ``run_experiment_suite.py``
(``results.json``) and ``run_corruption_eval.py`` (``corruption_results.json``
and the per-clip dumps in ``scores/``), plus the optional output of
``measure_resources.py``. Prints Markdown:

1. AUC by arm and condition, mean +- std over seeds;
2. paired difference to the baseline on clean inputs and averaged over the
   audio-corrupted conditions, with a 95% clip-level paired bootstrap
   interval (the same resampled clips for every seed), and the resource cost;
3. AUC under "audio missing" for clips whose forgery is visible in the video
   against clips with an audio-only forgery (report Section 5.4);
4. the gate value distribution on clean, audio-missing and video-missing
   inputs for the gated arms;
5. unfavourable cases: conditions where an arm is below the baseline, and
   clips the baseline gets right that the arm gets wrong.

    python scripts/analyze_improvement.py --suites runs/seed0 runs/seed1 runs/seed2 \\
        --resources results/resources.json --plot docs/improvement.svg
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import roc_auc_score

from rmfdn_ablation.config import FORGERY_TYPE_TO_TECHNIQUES
from rmfdn_ablation.corruptions import AUDIO_CORRUPTED, CONDITIONS

ARMS = ("base", "dropout", "gate", "gate_dropout")
ARM_LABEL = {"base": "Baseline", "dropout": "+ dropout", "gate": "+ gate",
             "gate_dropout": "+ gate + dropout"}


def auc(labels, scores) -> float:
    try:
        return float(roc_auc_score(labels, scores) * 100)
    except ValueError:
        return float("nan")


def load(suites, results_name, scores_name):
    """-> {arm: {seed: {"results": run entry, "scores": {condition: npz dict},
    "corruption": {condition: metrics}}}}"""
    data = {}
    for suite in map(Path, suites):
        training = json.load(open(suite / "results.json"))
        corruption = json.load(open(suite / results_name))
        for name, entry in corruption.items():
            arm = entry.get("variant", "base")
            seed = entry.get("seed")
            scores = {}
            for condition in entry["conditions"]:
                path = suite / scores_name / f"{name}__{condition}.npz"
                if path.exists():
                    with np.load(path) as npz:
                        scores[condition] = {k: npz[k] for k in npz.files}
            data.setdefault(arm, {})[seed] = {"results": training.get(name, {}),
                                              "corruption": entry["conditions"],
                                              "scores": scores}
    return data


def mean_std(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}"


def table_auc(data, conditions):
    lines = ["| Arm | " + " | ".join(conditions) + " |", "|---|" + "---:|" * len(conditions)]
    for arm in ARMS:
        if arm not in data:
            continue
        cells = [mean_std([run["corruption"].get(c, {}).get("AUC") for run in data[arm].values()])
                 for c in conditions]
        lines.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_bootstrap(data, arm, conditions, resamples, rng):
    """Mean over seeds (and conditions) of AUC(arm) - AUC(base), with a 95%
    clip-level paired bootstrap interval."""
    seeds = sorted(set(data[arm]) & set(data["base"]))
    pairs = [(data[arm][s]["scores"].get(c), data["base"][s]["scores"].get(c))
             for s in seeds for c in conditions]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return None
    count = len(pairs[0][0]["is_fake"])
    if any(len(a["is_fake"]) != count or len(b["is_fake"]) != count for a, b in pairs):
        raise ValueError("arms were scored on different clip sets")

    def statistic(index):
        return float(np.mean([auc(a["is_fake"][index], a["pfake"][index])
                              - auc(b["is_fake"][index], b["pfake"][index])
                              for a, b in pairs]))

    point = statistic(np.arange(count))
    samples = [statistic(rng.integers(0, count, count)) for _ in range(resamples)]
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return point, low, high


def table_differences(data, resources, resamples, rng):
    audio = [c for c in AUDIO_CORRUPTED]
    lines = ["| Arm | Δ clean AUC [95% CI] | Δ AUC, mean of audio-corrupted [95% CI] "
             "| Parameters | Latency (ms/clip) | Peak memory (GB) | Training time (h) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        if arm not in data:
            continue
        if arm == "base":
            clean = corrupted = "0"
        else:
            cells = []
            for conditions in (["clean"], audio):
                result = paired_bootstrap(data, arm, conditions, resamples, rng)
                cells.append("n/a" if result is None else
                             f"{result[0]:+.2f} [{result[1]:+.2f}, {result[2]:+.2f}]")
            clean, corrupted = cells
        res = (resources or {}).get(arm, {})
        params = f"{res['parameters'] / 1e6:.2f} M" if "parameters" in res else "n/a"
        latency = f"{res['latency_ms_per_clip']:.2f}" if "latency_ms_per_clip" in res else "n/a"
        runs = list(data[arm].values())
        memory = mean_std([(r["results"].get("resources") or {}).get("peak_memory_gb")
                           for r in runs])
        hours = mean_std([r["results"]["minutes"] / 60 if "minutes" in r["results"] else None
                          for r in runs])
        lines.append(f"| {ARM_LABEL[arm]} | {clean} | {corrupted} | {params} | {latency} "
                     f"| {memory} | {hours} |")
    return "\n".join(lines)


def table_forgery_groups(data, condition="audio_missing"):
    visual = {t for t, bits in FORGERY_TYPE_TO_TECHNIQUES.items() if bits[0] or bits[1]}
    audio_only = {t for t, bits in FORGERY_TYPE_TO_TECHNIQUES.items()
                  if t != "pristine" and not (bits[0] or bits[1])}
    lines = [f"| Arm | AUC, visible forgery ({condition}) | AUC, audio-only forgery ({condition}) |",
             "|---|---:|---:|"]
    for arm in ARMS:
        if arm not in data:
            continue
        cells = []
        for group in (visual, audio_only):
            values = []
            for run in data[arm].values():
                scores = run["scores"].get(condition)
                if scores is None:
                    continue
                keep = np.isin(scores["ftype"], list(group | {"pristine"}))
                values.append(auc(scores["is_fake"][keep], scores["pfake"][keep]))
            cells.append(mean_std(values))
        lines.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_gate(data):
    lines = ["| Arm | Condition | Mean gate | Std | 25% | 75% |", "|---|---|---:|---:|---:|---:|"]
    for arm in ("gate", "gate_dropout"):
        for condition in ("clean", "audio_missing", "video_missing"):
            values = [run["scores"][condition]["gate"] for run in data.get(arm, {}).values()
                      if condition in run["scores"] and "gate" in run["scores"][condition]]
            if not values:
                continue
            gate = np.concatenate(values)
            q1, q3 = np.percentile(gate, [25, 75])
            lines.append(f"| {ARM_LABEL[arm]} | {condition} | {gate.mean():.3f} "
                         f"| {gate.std():.3f} | {q1:.3f} | {q3:.3f} |")
    return "\n".join(lines) if len(lines) > 2 else "No gated runs with score dumps."


def unfavourable(data, conditions):
    lines = []
    for arm in ARMS[1:]:
        if arm not in data:
            continue
        for condition in conditions:
            arm_auc = [r["corruption"].get(condition, {}).get("AUC") for r in data[arm].values()]
            base_auc = [r["corruption"].get(condition, {}).get("AUC") for r in data["base"].values()]
            if None in arm_auc or None in base_auc or not arm_auc:
                continue
            if np.mean(arm_auc) < np.mean(base_auc):
                lines.append(f"- {ARM_LABEL[arm]} is below the baseline under `{condition}`: "
                             f"{np.mean(arm_auc):.2f} vs {np.mean(base_auc):.2f} AUC")
        flipped, total = 0, 0
        for seed, run in data[arm].items():
            base = data["base"].get(seed, {}).get("scores", {}).get("clean")
            mine = run["scores"].get("clean")
            if base is None or mine is None:
                continue
            base_right = (base["pfake"] > 0.5) == base["is_fake"].astype(bool)
            mine_wrong = (mine["pfake"] > 0.5) != mine["is_fake"].astype(bool)
            flipped += int((base_right & mine_wrong).sum())
            total += len(mine_wrong)
        if total:
            lines.append(f"- {ARM_LABEL[arm]}, clean: {flipped} of {total} clip-seed pairs "
                         f"are right for the baseline and wrong for this arm")
    return "\n".join(lines) if lines else "- none"


def plot(data, conditions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"base": "#2E3B4E", "dropout": "#2B6CB0", "gate": "#B8C0CC", "gate_dropout": "#4E9A88"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axis = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(conditions))
    for arm in ARMS:
        if arm not in data:
            continue
        values = np.array([[r["corruption"].get(c, {}).get("AUC", np.nan) for c in conditions]
                           for r in data[arm].values()], dtype=float)
        mean = np.nanmean(values, 0)
        std = np.nanstd(values, 0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
        axis.errorbar(x, mean, yerr=std, marker="o", capsize=3, color=colors[arm],
                      label=ARM_LABEL[arm])
    axis.set_xticks(x, [c.replace("_", "\n") for c in conditions])
    axis.set_ylabel("Detection AUC (%)")
    axis.set_title("AUC by test condition (mean ± std over seeds)", loc="left",
                   color="#2E3B4E", fontweight="bold")
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suites", nargs="+", required=True, help="one suite directory per seed")
    ap.add_argument("--results-name", default="corruption_results.json")
    ap.add_argument("--scores-name", default="scores")
    ap.add_argument("--resources", default=None, help="output of measure_resources.py")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--output", default=None, help="also write the Markdown here")
    args = ap.parse_args()

    data = load(args.suites, args.results_name, args.scores_name)
    if "base" not in data:
        sys.exit("no baseline arm found in the given suites")
    resources = json.load(open(args.resources)) if args.resources else None
    conditions = [c for c in CONDITIONS
                  if any(c in r["corruption"] for runs in data.values() for r in runs.values())]
    rng = np.random.default_rng(args.seed)
    seeds = sorted({s for runs in data.values() for s in runs})
    report = "\n\n".join([
        f"Seeds: {seeds}. Arms: {[a for a in ARMS if a in data]}.",
        "### AUC by condition (mean ± std over seeds)", table_auc(data, conditions),
        f"### Paired difference to the baseline ({args.bootstrap} bootstrap resamples) "
        "and resource cost", table_differences(data, resources, args.bootstrap, rng),
        "### Audio missing: visible against audio-only forgeries", table_forgery_groups(data),
        "### Gate values", table_gate(data),
        "### Unfavourable cases", unfavourable(data, conditions),
    ])
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n")
    if args.plot:
        plot(data, conditions, args.plot)
        print(f"\nwrote {args.plot}", file=sys.stderr)


if __name__ == "__main__":
    main()
