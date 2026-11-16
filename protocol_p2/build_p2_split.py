#!/usr/bin/env python3
"""Build the Protocol-2 identity split: gender-stratified, identity-disjoint.

Protocol 2 partitions the IDForge_v2 *identities* (no clip is shared across
splits, so train/val/test never see the same speaker) under two requirements:

1. **Gender stratification (hard constraint).** Each split holds an exact
   number of male / female identities::

       train = 24 M + 10 F   (34 ids)
       val   =  4 M +  2 F   ( 6 ids)
       test  = 10 M +  4 F   (14 ids)

2. **Shot-count balance (objective).** Per-identity shot counts are highly
   uneven, so *which* identity lands in *which* split decides how many shots
   each split ends up with. We want each split's shot total to mirror its
   identity share -- i.e. proportional to its identity count -- so the
   targets are ``total_shots * split_ids / total_ids``. The exact targets are
   usually unreachable (a handful of large identities), so we search for the
   assignment that minimises the deviation from them.

Because the per-split gender counts are fixed, every candidate move is a swap
of two *same-gender* identities between two splits: this preserves the gender
constraint automatically. We minimise the sum of squared shot-deviations with
multi-restart hill climbing -- at 54 identities this reaches the global (or
indistinguishably-near-global) optimum in well under a second.

Input  : a CSV with a header and columns ``identity,gender,shots``
         (gender in {M, F} / {male, female}; shots = integer clip count).
Output : a JSON split file plus a printed report (per-split gender counts,
         shot totals vs. targets, and absolute / relative deviation).

Usage::

    python protocol_p2/build_p2_split.py --metadata ids.csv --output p2_split.json
    python protocol_p2/build_p2_split.py --demo        # synthetic data, no input
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Split specification (identity cardinalities per split, per gender)
# --------------------------------------------------------------------------- #

SPLIT_NAMES = ("train", "val", "test")

#: Required identity count per (split, gender). Shot targets are derived from
#: the data (proportional to identity count), not hard-coded.
GENDER_CARDINALITY: Dict[str, Dict[str, int]] = {
    "train": {"M": 24, "F": 10},
    "val":   {"M": 4,  "F": 2},
    "test":  {"M": 10, "F": 4},
}


@dataclass(frozen=True)
class Identity:
    name: str
    gender: str   # "M" or "F"
    shots: int


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def _normalise_gender(value: str) -> str:
    v = value.strip().lower()
    if v in ("m", "male"):
        return "M"
    if v in ("f", "female"):
        return "F"
    raise ValueError(f"unrecognised gender label: {value!r} (use M/F or male/female)")


def load_identities(csv_path: str) -> List[Identity]:
    """Read ``identity,gender,shots`` rows into Identity records."""
    identities: List[Identity] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"identity", "gender", "shots"}
        missing = required - set(name.strip() for name in reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        for row in reader:
            identities.append(Identity(
                name=row["identity"].strip(),
                gender=_normalise_gender(row["gender"]),
                shots=int(row["shots"]),
            ))
    return identities


def validate_feasibility(identities: List[Identity]) -> None:
    """Fail early if the data cannot satisfy the gender cardinalities."""
    required = {
        gender: sum(spec[gender] for spec in GENDER_CARDINALITY.values())
        for gender in ("M", "F")
    }
    available = {
        gender: sum(1 for i in identities if i.gender == gender)
        for gender in ("M", "F")
    }
    if available != required:
        raise ValueError(
            f"gender counts do not match the split spec: "
            f"have {available}, need {required}")


# --------------------------------------------------------------------------- #
# Targets and objective
# --------------------------------------------------------------------------- #

def shot_targets(identities: List[Identity]) -> Dict[str, float]:
    """Per-split shot target = total_shots * (split_ids / total_ids)."""
    total_shots = sum(i.shots for i in identities)
    total_ids = len(identities)
    return {
        split: total_shots * (sum(GENDER_CARDINALITY[split].values()) / total_ids)
        for split in SPLIT_NAMES
    }


def _split_shot_totals(assignment: Dict[str, List[Identity]]) -> Dict[str, int]:
    return {split: sum(i.shots for i in members) for split, members in assignment.items()}


def _objective(assignment: Dict[str, List[Identity]], targets: Dict[str, float]) -> float:
    """Sum of squared shot-deviations from the targets (lower is better)."""
    totals = _split_shot_totals(assignment)
    return sum((totals[split] - targets[split]) ** 2 for split in SPLIT_NAMES)


# --------------------------------------------------------------------------- #
# Search: multi-restart hill climbing over gender-preserving swaps
# --------------------------------------------------------------------------- #

def _random_valid_assignment(identities: List[Identity],
                             rng: random.Random) -> Dict[str, List[Identity]]:
    """A random partition that already satisfies the gender cardinalities."""
    by_gender = {"M": [i for i in identities if i.gender == "M"],
                 "F": [i for i in identities if i.gender == "F"]}
    for pool in by_gender.values():
        rng.shuffle(pool)
    assignment: Dict[str, List[Identity]] = {split: [] for split in SPLIT_NAMES}
    for gender, pool in by_gender.items():
        cursor = 0
        for split in SPLIT_NAMES:
            count = GENDER_CARDINALITY[split][gender]
            assignment[split].extend(pool[cursor:cursor + count])
            cursor += count
    return assignment


def _hill_climb(assignment: Dict[str, List[Identity]], targets: Dict[str, float],
                rng: random.Random, max_iterations: int = 6000) -> float:
    """Greedily accept gender-preserving swaps that lower the objective."""
    current = _objective(assignment, targets)
    for _ in range(max_iterations):
        a, b = rng.sample(SPLIT_NAMES, 2)
        gender = rng.choice(("M", "F"))
        members_a = [i for i in assignment[a] if i.gender == gender]
        members_b = [i for i in assignment[b] if i.gender == gender]
        if not members_a or not members_b:
            continue
        ia, ib = rng.choice(members_a), rng.choice(members_b)

        # Effect of swapping ia<->b and ib<->a on the two affected splits.
        totals = _split_shot_totals(assignment)
        delta = ib.shots - ia.shots
        new_a = totals[a] + delta
        new_b = totals[b] - delta
        before = (totals[a] - targets[a]) ** 2 + (totals[b] - targets[b]) ** 2
        after = (new_a - targets[a]) ** 2 + (new_b - targets[b]) ** 2
        if after < before - 1e-9:
            assignment[a].remove(ia); assignment[a].append(ib)
            assignment[b].remove(ib); assignment[b].append(ia)
            current += after - before
    return current


def solve_split(identities: List[Identity], seed: int = 0,
                restarts: int = 400) -> Tuple[Dict[str, List[Identity]], Dict[str, float]]:
    """Return the best gender-stratified, shot-balanced assignment found."""
    targets = shot_targets(identities)
    rng = random.Random(seed)
    best_assignment, best_score = None, float("inf")
    for _ in range(restarts):
        candidate = _random_valid_assignment(identities, rng)
        score = _hill_climb(candidate, targets, rng)
        if score < best_score:
            best_assignment, best_score = candidate, score
    return best_assignment, targets


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_report(assignment: Dict[str, List[Identity]], targets: Dict[str, float]) -> None:
    totals = _split_shot_totals(assignment)
    total_shots = sum(totals.values())
    print(f"{'split':6s} {'M':>3s} {'F':>3s} {'ids':>4s} "
          f"{'shots':>9s} {'target':>9s} {'dev':>8s} {'dev%':>7s} {'share%':>7s}")
    for split in SPLIT_NAMES:
        members = assignment[split]
        males = sum(1 for i in members if i.gender == "M")
        females = sum(1 for i in members if i.gender == "F")
        deviation = totals[split] - targets[split]
        print(f"{split:6s} {males:3d} {females:3d} {len(members):4d} "
              f"{totals[split]:9d} {targets[split]:9.0f} {deviation:+8.0f} "
              f"{100 * deviation / targets[split]:+6.2f}% "
              f"{100 * totals[split] / total_shots:6.2f}%")
    print(f"{'total':6s} {'':3s} {'':3s} {sum(len(m) for m in assignment.values()):4d} "
          f"{total_shots:9d}")


def assignment_to_dict(assignment: Dict[str, List[Identity]]) -> dict:
    """Serialisable split: identity lists plus per-split summary statistics."""
    out = {"splits": {}, "summary": {}}
    for split in SPLIT_NAMES:
        members = sorted(assignment[split], key=lambda i: i.name)
        out["splits"][split] = [i.name for i in members]
        out["summary"][split] = {
            "ids": len(members),
            "male": sum(1 for i in members if i.gender == "M"),
            "female": sum(1 for i in members if i.gender == "F"),
            "shots": sum(i.shots for i in members),
        }
    return out


# --------------------------------------------------------------------------- #
# Synthetic data (for --demo, until the real per-identity table is supplied)
# --------------------------------------------------------------------------- #

def synthetic_identities(seed: int = 1) -> List[Identity]:
    """54 identities (38 M / 16 F), heavily skewed shots summing to 249,138."""
    rng = random.Random(seed)
    genders = ["M"] * 38 + ["F"] * 16
    # Log-normal-ish skew so a few identities dominate, like the real set.
    raw = [rng.lognormvariate(1.0, 0.9) for _ in genders]
    scale = 249_138 / sum(raw)
    shots = [max(1, round(value * scale)) for value in raw]
    # Nudge the rounding so the grand total is exactly 249,138.
    shots[0] += 249_138 - sum(shots)
    return [Identity(f"id{n:02d}", g, s) for n, (g, s) in enumerate(zip(genders, shots))]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Protocol-2 identity split")
    parser.add_argument("--metadata", help="CSV with columns identity,gender,shots")
    parser.add_argument("--output", help="write the split JSON here")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=400)
    parser.add_argument("--demo", action="store_true",
                        help="run on synthetic data (no --metadata needed)")
    args = parser.parse_args()

    if args.demo:
        identities = synthetic_identities()
        print("[demo] synthetic 54 identities (38 M / 16 F), total "
              f"{sum(i.shots for i in identities)} shots\n")
    elif args.metadata:
        identities = load_identities(args.metadata)
    else:
        parser.error("provide --metadata CSV or --demo")

    validate_feasibility(identities)
    assignment, targets = solve_split(identities, seed=args.seed, restarts=args.restarts)
    print_report(assignment, targets)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(assignment_to_dict(assignment), handle, indent=2)
        print(f"\nsplit written to {args.output}")


if __name__ == "__main__":
    main()
