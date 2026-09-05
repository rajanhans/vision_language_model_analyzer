"""Tiny JSON evaluation runner for repeatable VLM checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .agent import VLMAgent


def main() -> None:
    """Run each manifest case and exit non-zero when any expected term is missing."""
    parser = argparse.ArgumentParser(description="Run a basic VLM evaluation manifest.")
    parser.add_argument("manifest", type=Path, help="JSON file containing an array of cases")
    parser.add_argument("--model", help="Override VLM_MODEL")
    args = parser.parse_args()
    load_dotenv()

    # The manifest remains deliberately simple: each case names an image, question, and
    # optional terms that must occur in the model's lower-cased answer.
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    agent = VLMAgent(model=args.model)
    passed = 0
    for index, case in enumerate(cases, start=1):
        result = agent.run(case["image"], case["question"], case.get("detail", "auto"))
        answer_lower = result.answer.lower()
        expected = [term.lower() for term in case.get("expected_terms", [])]
        missing = [term for term in expected if term not in answer_lower]
        status = "PASS" if not missing else "FAIL"
        passed += not missing
        print(f"[{status}] {index}: {case.get('name', case['question'])}")
        if missing:
            print(f"  Missing expected terms: {missing}")

    print(f"\n{passed}/{len(cases)} cases passed")
    raise SystemExit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()

