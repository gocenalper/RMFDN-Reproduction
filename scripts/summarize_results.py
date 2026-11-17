#!/usr/bin/env python3
"""Print the stored test results as a Markdown table; optionally plot them.

Reads every ``results/<group>/<cell>/test_results.json`` (and the matching
``results.json`` for the training time) and prints one row per run. Nothing is
typed by hand: the README tables and ``docs/results.svg`` come from this script.

    python scripts/summarize_results.py
    python scripts/summarize_results.py --plot docs/results.svg   # needs matplotlib
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Numbers published by Xu et al. (arXiv:2401.11764v3, Tables 2 and 3), in
#: percent: official split, identity reference on, 90,000 iterations.
PAPER = {
    "VA": {"ACC": 90.63, "AUC": 98.12, "mAP": 86.67},
    "VAT": {"ACC": 92.90, "AUC": 98.40, "mAP": 93.38},
}

MODALITY_ORDER = ("VA", "VAT")
SHORT = {"lip_sync": "lip", "face_swap": "face", "tts": "tts", "rvc": "rvc",
         "audio_shuffle": "a-shuf", "llm_textgen": "llm", "text_shuffle": "t-shuf"}


def load_runs(results_root: Path) -> list:
    runs = []
    for test_path in sorted(results_root.glob("*/*/test_results.json")):
        training_path = test_path.with_name("results.json")
        training = json.loads(training_path.read_text()) if training_path.exists() else {}
        for name, entry in json.loads(test_path.read_text()).items():
            runs.append({
                "group": test_path.parent.parent.name,
                "name": name,
                "modality": entry["spec"]["modality"],
                "identity": entry["spec"]["identity"],
                "protocol": entry["spec"]["protocol"],
                "seed": entry["config"].get("seed"),
                "iters": entry["config"].get("total_iters"),
                "minutes": training.get(name, {}).get("minutes"),
                "test": entry["test"],
            })
    runs.sort(key=lambda run: (run["group"], MODALITY_ORDER.index(run["modality"])))
    return runs


def markdown_tables(runs: list) -> str:
    lines = ["| Run | Modalities | ACC | AUC | mAP | CF1 | OF1 | Bits scored | Train time |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for run in runs:
        test = run["test"]
        bits = ", ".join(SHORT[b] for b in test["metric_bits"])
        minutes = f"{run['minutes']:.0f} min" if run["minutes"] is not None else "n/a"
        lines.append(
            f"| `{run['name']}` | {run['modality']} | {test['ACC']:.2f} | {test['AUC']:.2f} "
            f"| {test['mAP']:.2f} | {test['CF1']:.2f} | {test['OF1']:.2f} | {bits} | {minutes} |")
    lines += ["", "| Run | " + " | ".join(SHORT.values()) + " |",
              "|---|" + "---:|" * len(SHORT)]
    for run in runs:
        per_bit = run["test"].get("per_bit_ap", {})
        cells = [f"{per_bit[bit]:.1f}" if bit in per_bit else "–" for bit in SHORT]
        lines.append(f"| `{run['name']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(runs: list, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teal, blue, navy, gray = "#4E9A88", "#2B6CB0", "#2E3B4E", "#B8C0CC"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4),
                                         gridspec_kw={"width_ratios": [1, 1.25]})

    # Left: detection AUC, this repository against the published numbers.
    labels = [run["modality"] for run in runs]
    positions = range(len(runs))
    mine = [run["test"]["AUC"] for run in runs]
    paper = [PAPER.get(run["modality"], {}).get("AUC", 0) for run in runs]
    left.bar([p - 0.2 for p in positions], paper, 0.4, color=gray, label="Paper (official split, with reference)")
    left.bar([p + 0.2 for p in positions], mine, 0.4, color=teal,
             label="This repo (identity-disjoint, no reference)")
    for p, (a, b) in enumerate(zip(paper, mine)):
        left.text(p - 0.2, a + 1, f"{a:.1f}", ha="center", fontsize=9, color=navy)
        left.text(p + 0.2, b + 1, f"{b:.1f}", ha="center", fontsize=9, color=navy)
    left.set_xticks(list(positions), labels)
    left.set_ylim(50, 108)
    left.set_ylabel("Detection AUC (%)")
    left.set_title("Real / fake detection", loc="left", color=navy, fontweight="bold")
    left.legend(frameon=False, fontsize=9, loc="upper left")

    # Right: per-technique average precision of the full model.
    full = next((run for run in runs if run["modality"] == "VAT"), runs[-1])
    per_bit = full["test"]["per_bit_ap"]
    names = [SHORT[bit] for bit in per_bit]
    values = list(per_bit.values())
    colors = [blue if bit in ("lip_sync", "face_swap") else
              teal if bit in ("tts", "rvc", "audio_shuffle") else navy for bit in per_bit]
    right.barh(names[::-1], values[::-1], color=colors[::-1])
    for index, value in enumerate(values[::-1]):
        right.text(value + 0.6, index, f"{value:.1f}", va="center", fontsize=9, color=navy)
    right.set_xlim(50, 106)
    right.set_xlabel("Average precision (%)")
    right.set_title(f"Technique attribution, {full['modality']} model",
                    loc="left", color=navy, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (blue, teal, navy)]
    right.legend(handles, ["visual", "audio", "text"], frameon=False, fontsize=9,
                 loc="lower right")

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white")
    print(f"wrote {output}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default=str(REPO_ROOT / "results"))
    parser.add_argument("--plot", default=None, help="also write a figure here (needs matplotlib)")
    args = parser.parse_args()
    runs = load_runs(Path(args.results))
    if not runs:
        sys.exit(f"no test_results.json found under {args.results}")
    print(markdown_tables(runs))
    if args.plot:
        plot(runs, Path(args.plot))


if __name__ == "__main__":
    main()
