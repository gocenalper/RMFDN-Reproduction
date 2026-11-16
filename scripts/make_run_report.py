#!/usr/bin/env python3
"""One-page PDF summary for a completed single-factor experiment.

Every number on the page is read from artifacts on disk -- the plan file, the
suite results.json and (when present) the test-eval JSON. Nothing is typed in
by hand and missing values render as an em dash, never as a guess. Next to the
PDF a ``report_data.json`` sidecar records the same numbers plus the sha256 of
each source, so prose written later can be audited with check_claims.py.
A row is upserted into ``reports/INDEX.csv`` keyed by exp_id.

Usage (matplotlib is not in the project venv; use uv's ephemeral install):

    uv run --with matplotlib scripts/make_run_report.py \
        --plan experiments/plans/E001_peak_lr_1e-4.json \
        [--test-eval path/to/test_eval.json] \
        [--baseline-test path/to/test_eval.json] [--out reports]

On Colab plain ``python scripts/make_run_report.py ...`` works (mpl ships).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_plan_check import run_name_for  # noqa: E402

METRIC_ROWS = ("ACC", "AUC", "mAP", "CF1", "OF1")
EM_DASH = "—"


def sha12(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except Exception:
        return "unknown"


def load_run_record(ref: str | None, sources: list) -> dict | None:
    """Full raw record for a reference.

    ``path::run`` reads that record; a plan-file path follows the plan to its
    recorded results.json (metrics stay None when the plan has no results yet).
    """
    if not ref:
        return None
    if "::" not in ref:
        plan = json.load(open(ref))
        sources.append({"path": ref, "sha256_12": sha12(ref)})
        results_json = (plan.get("results") or {}).get("results_json")
        if not results_json:
            return {"spec": plan["spec"]}
        ref = f"{results_json}::{run_name_for(plan['spec'])}"
    path, run = ref.split("::", 1)
    record = json.load(open(path)).get(run)
    if record:
        sources.append({"path": path, "sha256_12": sha12(path)})
    return record


def load_test_metrics(path: str | None, run_name: str, sources: list) -> dict | None:
    """Pull record[run]["test"] out of a test-eval (or suite) JSON file."""
    if not path:
        return None
    records = json.load(open(path))
    record = records.get(run_name)
    if not record or not record.get("test"):
        return None
    sources.append({"path": path, "sha256_12": sha12(path)})
    return record["test"]


def numbers_for(record: dict | None, test_metrics: dict | None) -> dict:
    """Collect the reportable numbers for one side (candidate or baseline)."""
    out = {"best_val": None, "minutes": None, "per_bit_acc": None}
    if record:
        out["best_val"] = record.get("best_val_score")
        out["minutes"] = record.get("minutes")
        if not test_metrics and record.get("test"):
            test_metrics = record["test"]
    for key in METRIC_ROWS:
        out[key] = None if not test_metrics else test_metrics.get(key)
    if test_metrics:
        out["per_bit_acc"] = test_metrics.get("per_bit_acc")
    return out


def shorten(text: str, width: int) -> str:
    return text if len(text) <= width else "…" + text[-(width - 1):]


def fmt(value, nd=2) -> str:
    if value is None:
        return EM_DASH
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def resolve_metric(path: str, numbers: dict):
    """'test.AUC' -> numbers['AUC']; 'best_val_score'/'val.best' -> best_val."""
    leaf = path.split(".")[-1]
    if leaf in numbers:
        return numbers[leaf]
    if "val" in path:
        return numbers.get("best_val")
    return None


def verdict_for(plan: dict, cand: dict, base: dict) -> tuple[str, float | None]:
    metric = plan.get("success_metric", "")
    c = resolve_metric(metric, cand)
    b = resolve_metric(metric, base)
    if c is None or b is None:
        return "INCONCLUSIVE (metric missing)", None
    delta = c - b
    direction = plan.get("expected_direction", "up")
    supported = delta > 0 if direction == "up" else delta < 0
    return ("SUPPORTED" if supported else "NOT SUPPORTED"), delta


def upsert_index(out_root: Path, row: dict) -> None:
    index = out_root / "INDEX.csv"
    fields = list(row)
    rows = []
    if index.exists():
        with open(index) as handle:
            rows = [r for r in csv.DictReader(handle) if r["exp_id"] != row["exp_id"]]
    rows.append({k: row.get(k, "") for k in fields})
    with open(index, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_pdf(path: Path, plan: dict, run_name: str, cand: dict, base: dict,
               verdict: str, delta, sources: list, head: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    factor = plan["factor"]
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    mono = {"family": "monospace"}

    fig.text(0.06, 0.965, plan["exp_id"], size=17, weight="bold")
    fig.text(0.94, 0.965, f"{datetime.date.today().isoformat()}   git {head}",
             size=9, ha="right", color="0.35")
    fig.text(0.06, 0.945,
             f"run {run_name}   baseline {shorten(plan['baseline'], 78)}",
             size=8.5, color="0.35", **mono)

    fig.text(0.06, 0.912, "FACTOR", size=9, weight="bold", color="0.4")
    fig.text(0.06, 0.892,
             f"{factor['key']}: {factor.get('baseline_value', '?')} → "
             f"{factor.get('value', '?')}", size=13, weight="bold")
    hypothesis = textwrap.fill(str(plan.get("hypothesis", "")), 108)
    fig.text(0.06, 0.874, hypothesis, size=8.5, va="top", color="0.25")

    fig.text(0.06, 0.826, f"VERDICT ({plan.get('success_metric', '?')}, "
                          f"expected {plan.get('expected_direction', 'up')}): ",
             size=9, weight="bold", color="0.4")
    verdict_text = verdict if delta is None else f"{verdict}   Δ = {delta:+.2f}"
    fig.text(0.40, 0.826, verdict_text, size=11, weight="bold",
             color={"SUPPORTED": "#1a7a3a"}.get(verdict, "#8a1a1a"))
    fig.text(0.06, 0.810, "single seed — point estimate, no variance claim",
             size=7.5, style="italic", color="0.45")

    lines = [f"{'metric':<10}{'baseline':>12}{'this run':>12}{'delta':>10}",
             "-" * 44]
    for key in (*METRIC_ROWS, "best_val", "minutes"):
        c, b = cand.get(key), base.get(key)
        d = f"{c - b:+.2f}" if isinstance(c, (int, float)) and isinstance(b, (int, float)) else EM_DASH
        lines.append(f"{key:<10}{fmt(b):>12}{fmt(c):>12}{d:>10}")
    fig.text(0.06, 0.775, "\n".join(lines), size=9.5, va="top", **mono)

    bits_c, bits_b = cand.get("per_bit_acc") or {}, base.get("per_bit_acc") or {}
    if bits_c:
        ax = fig.add_axes([0.16, 0.40, 0.48, 0.20])
        names = list(bits_c)
        y = range(len(names))
        ax.barh([i + 0.18 for i in y], [bits_c[n] for n in names], 0.36,
                label="this run", color="#2a6fb0")
        if bits_b:
            ax.barh([i - 0.18 for i in y], [bits_b.get(n, 0) for n in names],
                    0.36, label="baseline", color="#b0b0b0")
        ax.set_yticks(list(y), names, size=8)
        ax.set_xlim(0, 100)
        ax.set_title("per-bit accuracy (test)", size=9, loc="left")
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="lower right")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    config = plan["config"]
    items = [f"{'>' if k == factor['key'] else ' '} {k} = {config[k]!r}"
             for k in sorted(config)]
    half = (len(items) + 1) // 2
    fig.text(0.06, 0.355, "effective config (> = the factor)", size=8,
             weight="bold", color="0.4")
    fig.text(0.06, 0.340, "\n".join(items[:half]), size=6.4, va="top", **mono)
    fig.text(0.53, 0.340, "\n".join(items[half:]), size=6.4, va="top", **mono)

    footer = ["sources (sha256/12):"] + \
             [f"  {s['sha256_12']}  {shorten(s['path'], 96)}" for s in sources] + \
             ["generated by scripts/make_run_report.py -- every number above "
              "is read from these files; missing = " + EM_DASH]
    fig.text(0.06, 0.075, "\n".join(footer), size=7, va="top", color="0.4", **mono)

    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--test-eval", default=None,
                        help="test-eval JSON for the candidate (default: "
                             "plan.results.test_eval_json)")
    parser.add_argument("--baseline-test", default=None,
                        help="test-eval JSON holding the baseline run's test "
                             "metrics (default: read inline from the baseline ref)")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    plan = json.load(open(args.plan))
    run_name = run_name_for(plan["spec"])
    sources = [{"path": args.plan, "sha256_12": sha12(args.plan)}]

    results_json = (plan.get("results") or {}).get("results_json")
    cand_train = load_run_record(f"{results_json}::{run_name}", sources) \
        if results_json else None
    test_path = args.test_eval or (plan.get("results") or {}).get("test_eval_json")
    cand_test = load_test_metrics(test_path, run_name, sources)
    cand = numbers_for(cand_train, cand_test)

    base_record = load_run_record(plan["baseline"], sources)
    base_run = (run_name_for(base_record["spec"])
                if base_record and base_record.get("spec") else run_name)
    base_test = load_test_metrics(args.baseline_test, base_run, sources)
    base = numbers_for(base_record, base_test)

    verdict, delta = verdict_for(plan, cand, base)
    head = git_head()

    out_dir = Path(args.out) / plan["exp_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "report.pdf"
    render_pdf(pdf_path, plan, run_name, cand, base, verdict, delta, sources, head)

    sidecar = {
        "exp_id": plan["exp_id"], "generated": datetime.datetime.now().isoformat(),
        "git_head": head, "run_name": run_name,
        "factor": plan["factor"], "baseline": plan["baseline"],
        "success_metric": plan.get("success_metric"),
        "verdict": verdict, "delta": delta,
        "candidate": {k: v for k, v in cand.items() if k != "per_bit_acc"},
        "baseline_metrics": {k: v for k, v in base.items() if k != "per_bit_acc"},
        "per_bit_acc": {"candidate": cand.get("per_bit_acc"),
                        "baseline": base.get("per_bit_acc")},
        "sources": sources,
    }
    json.dump(sidecar, open(out_dir / "report_data.json", "w"), indent=1)

    upsert_index(Path(args.out), {
        "exp_id": plan["exp_id"], "date": datetime.date.today().isoformat(),
        "run_name": run_name, "factor": f"{plan['factor']['key']}="
        f"{plan['factor'].get('value')}", "baseline": plan["baseline"],
        **{key: fmt(cand.get(key)) for key in METRIC_ROWS},
        "best_val": fmt(cand.get("best_val")), "minutes": fmt(cand.get("minutes")),
        "verdict": verdict, "report": str(pdf_path),
    })
    print(f"report : {pdf_path}")
    print(f"sidecar: {out_dir / 'report_data.json'}")
    print(f"index  : {Path(args.out) / 'INDEX.csv'} (upserted {plan['exp_id']})")


if __name__ == "__main__":
    main()
