#!/usr/bin/env python3
"""Deprecated compatibility guard for the former circular-labeling script.

EDA will be implemented as a separate task. This file intentionally performs no
label inference and writes no dataset.
"""

import sys

MESSAGE = (
    "scripts/eda_and_labeling.py is deprecated because it inferred target labels "
    "from model features. Use scripts/label_dataset.py with an independently "
    "curated label manifest. A separate EDA command will be added in Task 5.1B."
)


def main() -> int:
    """Return a failure code so stale automation cannot create circular labels."""
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())