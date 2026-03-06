#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    mnist_cmd = [
        "python3",
        str(root / "scripts" / "run_mnist_oneshot.py"),
        "--unsupervised-subset",
        "60000",
        "--aug-per-class",
        "0",
    ]
    breakout_cmd = [
        "python3",
        str(root / "scripts" / "run_breakout_eval.py"),
        "--episodes",
        "2",
        "--max-steps",
        "12000",
    ]
    rc1 = subprocess.call(mnist_cmd, env=env)
    rc2 = subprocess.call(breakout_cmd, env=env)
    print(json.dumps({"mnist_rc": rc1, "breakout_rc": rc2}, indent=2))


if __name__ == "__main__":
    main()
