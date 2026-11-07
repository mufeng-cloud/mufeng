"""Polymer-group holdout and group-disjoint development folds; no model code."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def build_group_split_plan(polymer_values, seed, test_fraction=0.2, n_splits=5):
    """Split using input polymer identity only; never inspect target labels."""
    groups = np.asarray(list(polymer_values), dtype=object)
    if any(not isinstance(value, str) or not value.strip() for value in groups):
        raise ValueError("Every record must have a nonempty polymer identity.")
    n_groups = len(set(groups.tolist()))
    n_test_groups = math.ceil(n_groups * test_fraction)
    if n_groups - n_test_groups < n_splits:
        raise ValueError("Too few development polymer types for the requested group folds.")

    outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    development, test = next(outer.split(np.zeros(len(groups)), groups=groups))
    development = np.sort(development)
    test = np.sort(test)
    development_groups = groups[development]
    inner = GroupKFold(n_splits=n_splits)
    folds = []
    for fold, (train_relative, validation_relative) in enumerate(
            inner.split(np.zeros(len(development)), groups=development_groups)):
        folds.append({
            "fold": fold,
            "train_indices": development[train_relative].tolist(),
            "validation_indices": development[validation_relative].tolist(),
        })
    return {
        "seed": int(seed),
        "group_column": "polymer",
        "group_identity": "exact polymer input string",
        "test_fraction_of_polymer_types": float(test_fraction),
        "n_polymer_types": n_groups,
        "n_test_polymer_types": len(set(groups[test].tolist())),
        "test_indices": test.tolist(),
        "development_indices": development.tolist(),
        "n_splits": int(n_splits),
        "folds": folds,
    }


def save_group_split_plan(plan, data_csv, output_directory):
    """Save the actual runtime split as metadata, without exporting split CSVs."""
    record = dict(plan)
    path = Path(data_csv)
    record["data_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["data_filename"] = path.name
    destination = Path(output_directory) / "polymer_group_split.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
