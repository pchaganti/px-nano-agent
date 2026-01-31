#!/usr/bin/env python3
"""Run all end-to-end tests.

Usage:
    uv run python e2e/run_all.py

These tests make real API calls and are not part of the regular test suite.
"""

import asyncio
import importlib
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

E2E_TESTS = [
    "test_executor_cancellation",
]


async def run_all():
    print("=" * 60)
    print("Running End-to-End Tests")
    print("=" * 60)
    print()

    results = []

    for test_name in E2E_TESTS:
        print(f"Running: {test_name}")
        print("-" * 40)

        try:
            module = importlib.import_module(test_name)
            exit_code = await module.main()
            results.append((test_name, exit_code == 0))
        except Exception as e:
            print(f"Failed to run {test_name}: {e}")
            results.append((test_name, False))

        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed

    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {status}: {name}")

    print()
    print(f"Total: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(asyncio.run(run_all()))
