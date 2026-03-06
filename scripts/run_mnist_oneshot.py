#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cheap_universal_agi.config import BlueprintConfig
from cheap_universal_agi.mnist_oneshot import MnistOneShotSystem


def main():
    parser = argparse.ArgumentParser(description="Run MNIST one-shot benchmark")
    parser.add_argument("--unsupervised-subset", type=int, default=60000)
    parser.add_argument("--aug-per-class", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/mnist_oneshot_results.json"))
    args = parser.parse_args()

    cfg = BlueprintConfig()
    system = MnistOneShotSystem(cfg)
    _, y_test, test_z = system.fit(
        unsupervised_subset=args.unsupervised_subset,
        aug_per_class=args.aug_per_class,
    )
    result = system.evaluate(y_test=y_test, test_z=test_z)
    payload = {
        "accuracy": result.accuracy,
        "n_test": result.n_test,
        "per_class_accuracy": result.per_class_accuracy,
        "one_shot_indices": result.one_shot_indices,
        "confusion_matrix": result.confusion.tolist(),
        "unsupervised_subset": args.unsupervised_subset,
        "aug_per_class": args.aug_per_class,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"accuracy": result.accuracy, "n_test": result.n_test}, indent=2))


if __name__ == "__main__":
    main()
