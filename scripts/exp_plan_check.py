#!/usr/bin/env python3
"""Single-factor experiment control for the R-MFDN ablation.

Enforces the rule that every experiment changes exactly ONE controlled
parameter relative to its declared baseline. It runs at three moments:

  check   before launch: plan file vs. its declared baseline record
  verify  after the run: what actually ran (results.json embeds the true
          effective config) vs. what the plan promised
  diff    ad-hoc: classified diff between any two run references

A *run reference* is either
  path/to/results.json::run_name   -- suite results or test-eval output record
  path/to/plan.json                -- a plan file under experiments/plans/

Config keys are classified:
  runtime      never experimental, ignored (num_workers, log_interval, ...)
  environment  reported as WARN only (bert_model / ast_checkpoint paths differ
               between offline boxes without changing the experiment)
  experimental everything else, including the spec axes (spec.modality, ...)

batch_size and grad_accum_steps fold into one synthetic key
``effective_batch`` (= batch_size * grad_accum_steps): repacking the same
effective batch for a smaller GPU is environment, not an experiment.

Exit code 0 = PASS, 1 = FAIL, so launch cells can gate on it. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

RUNTIME_KEYS = {
    "num_workers", "log_interval", "skip_test_eval", "device",
}
ENVIRONMENT_KEYS = {
    "bert_model", "ast_checkpoint", "bert_mini_model", "bert_small_model",
    "bert_medium_model",
}
SPEC_AXES = ("cross_modal", "identity", "protocol", "modality")
PLAN_REQUIRED = ("exp_id", "baseline", "factor", "hypothesis",
                 "success_metric", "spec", "config")


def load_ref(ref: str) -> dict:
    """Resolve a run reference to {label, spec, config}."""
    if "::" in ref:
        path, run_name = ref.split("::", 1)
        records = json.load(open(path))
        if run_name not in records:
            sys.exit(f"FAIL: run {run_name!r} not found in {path} "
                     f"(has: {', '.join(sorted(records))})")
        record = records[run_name]
        if "config" not in record or "spec" not in record:
            sys.exit(f"FAIL: record {run_name!r} in {path} has no config/spec "
                     "(not_implemented cell, or a crashed run?)")
        return {"label": ref, "spec": record["spec"], "config": record["config"]}
    plan = json.load(open(ref))
    return {"label": f"{ref} (plan {plan.get('exp_id', '?')})",
            "spec": plan["spec"], "config": plan["config"], "plan": plan}


def run_name_for(spec: dict) -> str:
    return "_".join(spec[axis] for axis in SPEC_AXES)


def _values_equal(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def diff_refs(base: dict, cand: dict):
    """Classified diff. Returns (experimental, environment, notes) lists of
    (key, base_value, cand_value)."""
    experimental, environment, notes = [], [], []

    for axis in SPEC_AXES:
        b, c = base["spec"].get(axis), cand["spec"].get(axis)
        if b != c:
            experimental.append((f"spec.{axis}", b, c))

    b_cfg, c_cfg = dict(base["config"]), dict(cand["config"])

    def eff_batch(cfg):
        bs = cfg.pop("batch_size", None)
        ga = cfg.pop("grad_accum_steps", 1)
        return (None if bs is None else bs * ga), bs, ga

    b_eff, b_bs, b_ga = eff_batch(b_cfg)
    c_eff, c_bs, c_ga = eff_batch(c_cfg)
    if b_eff != c_eff:
        experimental.append(("effective_batch", b_eff, c_eff))
    elif (b_bs, b_ga) != (c_bs, c_ga):
        notes.append(f"batch repacked {b_bs}x{b_ga} -> {c_bs}x{c_ga} "
                     "(same effective batch: environment, not a factor)")

    for key in sorted(set(b_cfg) | set(c_cfg)):
        if key in RUNTIME_KEYS or key.startswith("_"):
            continue
        b, c = b_cfg.get(key, "<absent>"), c_cfg.get(key, "<absent>")
        if _values_equal(b, c):
            continue
        (environment if key in ENVIRONMENT_KEYS else experimental).append((key, b, c))

    return experimental, environment, notes


def print_diff(experimental, environment, notes, base_label, cand_label):
    print(f"baseline : {base_label}")
    print(f"candidate: {cand_label}")
    for key, b, c in experimental:
        print(f"  CHANGE  {key}: {b!r} -> {c!r}")
    for key, b, c in environment:
        print(f"  ENV     {key}: {b!r} -> {c!r}  (warn only)")
    for note in notes:
        print(f"  NOTE    {note}")
    if not (experimental or environment or notes):
        print("  (no differences)")


def _fail(message: str):
    print(f"FAIL: {message}")
    sys.exit(1)


def check_plan(plan_path: str) -> None:
    """Pre-launch gate: the plan changes exactly its one declared factor."""
    plan = json.load(open(plan_path))
    for field in PLAN_REQUIRED:
        if not plan.get(field):
            _fail(f"plan is missing required field {field!r}")
    if not str(plan["hypothesis"]).strip():
        _fail("hypothesis is empty")

    exp_id = plan["exp_id"]
    prefix = exp_id.split("_", 1)[0]
    plans_dir = Path(plan_path).resolve().parent
    twins = [p.name for p in plans_dir.glob(f"{prefix}_*.json")
             if p.resolve() != Path(plan_path).resolve()]
    if twins:
        _fail(f"exp id prefix {prefix} already used by: {', '.join(twins)}")

    base = load_ref(plan["baseline"])
    cand = {"label": f"plan {exp_id}", "spec": plan["spec"],
            "config": plan["config"]}
    experimental, environment, notes = diff_refs(base, cand)
    print_diff(experimental, environment, notes, base["label"], cand["label"])

    factor = plan["factor"]
    if len(experimental) != 1:
        _fail(f"{len(experimental)} experimental changes; exactly 1 allowed. "
              "Split this into one plan per factor.")
    key, b_val, c_val = experimental[0]
    if key != factor.get("key"):
        _fail(f"the changed key is {key!r} but the plan declares factor "
              f"{factor.get('key')!r}")
    if "value" in factor and not _values_equal(factor["value"], c_val):
        _fail(f"factor.value {factor['value']!r} != planned config value {c_val!r}")
    if "baseline_value" in factor and not _values_equal(factor["baseline_value"], b_val):
        _fail(f"factor.baseline_value {factor['baseline_value']!r} != "
              f"baseline's actual value {b_val!r}")
    print(f"PASS: single factor {key}: {b_val!r} -> {c_val!r}")


def verify_plan(plan_path: str, actual_ref: str | None) -> None:
    """Post-run gate: what ran is what the plan promised."""
    plan = json.load(open(plan_path))
    if actual_ref is None:
        results_json = (plan.get("results") or {}).get("results_json")
        if not results_json:
            _fail("plan.results.results_json is empty; pass --actual "
                  "path/to/results.json::run_name")
        actual_ref = f"{results_json}::{run_name_for(plan['spec'])}"
    actual = load_ref(actual_ref)
    planned = {"label": f"plan {plan['exp_id']}", "spec": plan["spec"],
               "config": plan["config"]}
    experimental, environment, notes = diff_refs(planned, actual)
    print_diff(experimental, environment, notes, planned["label"], actual["label"])
    if experimental:
        _fail("the run's effective config drifted from the plan "
              "(ad-hoc edit on the training box?). Either re-run, or record "
              "the drift as a new plan -- do not report this run under "
              f"{plan['exp_id']}.")
    print(f"PASS: {actual['label']} matches plan {plan['exp_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="pre-launch: plan vs. its baseline")
    p_check.add_argument("plan")

    p_verify = sub.add_parser("verify", help="post-run: plan vs. actual record")
    p_verify.add_argument("plan")
    p_verify.add_argument("--actual", default=None,
                          help="results.json::run_name (default: from the plan)")

    p_diff = sub.add_parser("diff", help="classified diff of two references")
    p_diff.add_argument("baseline")
    p_diff.add_argument("candidate")

    args = parser.parse_args()
    if args.cmd == "check":
        check_plan(args.plan)
    elif args.cmd == "verify":
        verify_plan(args.plan, args.actual)
    else:
        base, cand = load_ref(args.baseline), load_ref(args.candidate)
        print_diff(*diff_refs(base, cand), base["label"], cand["label"])


if __name__ == "__main__":
    main()
