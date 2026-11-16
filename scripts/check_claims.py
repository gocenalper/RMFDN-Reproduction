#!/usr/bin/env python3
"""Verify that every number in a written report exists in the metric sources.

The anti-hallucination gate for prose: any numeric claim in a draft (markdown,
tex, plain text) must be traceable to an actual artifact on disk
(results.json, a test-eval JSON, or a report_data.json sidecar). Numbers are
harvested recursively from the JSON sources; each numeric token in the text
must match one of them within its own displayed rounding, either directly or
as a x100 / /100 rescale (metrics are stored on the 0-100 scale, prose may
cite 0.924 or 92.4%).

Known limits (documented, not hidden): plain small integers (< 100) and years
1900-2100 are skipped by default because they are usually section numbers or
dates -- pass --strict to check them too. A PASS therefore means "no float or
large-integer claim is unsourced", not "the prose is correct".

Usage:
    python3 scripts/check_claims.py draft.md results.json test_eval.json ...

Exit 0 = every checked number is sourced; 1 = at least one is not. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

NUMBER = re.compile(r"(?<![\w.\-])-?\d+(?:\.\d+)?(?![\w.])")


def harvest(node, pool: set) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        pool.add(float(node))
    elif isinstance(node, dict):
        for value in node.values():
            harvest(value, pool)
    elif isinstance(node, list):
        for value in node:
            harvest(value, pool)


def decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


def sourced(token: str, pool: set) -> bool:
    value = float(token)
    tolerance = 0.5 * 10 ** -decimals(token)
    for candidate in (value, value * 100.0, value / 100.0):
        for known in pool:
            if abs(known - candidate) <= tolerance + 1e-12:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("text", help="draft file to audit")
    parser.add_argument("sources", nargs="+", help="JSON artifacts of record")
    parser.add_argument("--strict", action="store_true",
                        help="also check small integers and year-like numbers")
    args = parser.parse_args()

    pool: set = set()
    for source in args.sources:
        harvest(json.load(open(source)), pool)
    if not pool:
        sys.exit("FAIL: sources contain no numbers")

    text = open(args.text).read()
    unsourced = []
    for match in NUMBER.finditer(text):
        token = match.group(0)
        value = float(token)
        if not args.strict:
            if decimals(token) == 0 and (abs(value) < 100 or 1900 <= value <= 2100):
                continue
        if sourced(token, pool):
            continue
        start = max(0, match.start() - 30)
        context = text[start:match.end() + 30].replace("\n", " ")
        unsourced.append((token, context))

    if unsourced:
        print(f"FAIL: {len(unsourced)} number(s) not found in "
              f"{len(args.sources)} source file(s):")
        for token, context in unsourced:
            print(f"  {token:>12}  ...{context}...")
        print("Fix the prose or add the artifact that backs the claim; "
              "never adjust an artifact to match prose.")
        sys.exit(1)
    print(f"PASS: every checked number is backed by the sources "
          f"({len(pool)} source values).")


if __name__ == "__main__":
    main()
