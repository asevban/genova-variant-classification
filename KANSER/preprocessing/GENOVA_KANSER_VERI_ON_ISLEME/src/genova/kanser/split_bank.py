"""KANSER için sabit 5x10 dış + 4-fold iç grup-güvenli split bankası."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
META_PATH = ROOT / "data" / "processed" / "kanser" / "00_KANSER_TARGET_AND_METADATA.csv"
SPLIT_DIR = ROOT / "data" / "splits" / "kanser"
N_OUTER_SPLITS = 5
N_REPEATS = 10
N_INNER_SPLITS = 4
BASE_SEED = 4200


def _group_table(df: pd.DataFrame) -> list[dict]:
    groups = []
    for group_id, block in df.groupby("duplicate_group_id", sort=False):
        groups.append({
            "group_id": str(group_id),
            "ids": block["Variant_ID"].astype(str).tolist(),
            "n0": int((block["Label"] == 0).sum()),
            "n1": int((block["Label"] == 1).sum()),
            "size": int(len(block)),
        })
    return groups


def group_stratified_folds(df: pd.DataFrame, n_splits: int, seed: int) -> list[dict]:
    """Etiket ve grup bütünlüğünü birlikte dengeleyen deterministik greedy splitter."""
    rng = np.random.default_rng(seed)
    groups = _group_table(df)
    random_keys = {g["group_id"]: float(rng.random()) for g in groups}
    groups.sort(key=lambda g: (-g["size"], -max(g["n0"], g["n1"]), random_keys[g["group_id"]]))

    total = np.array([sum(g["n0"] for g in groups), sum(g["n1"] for g in groups)], dtype=float)
    target = total / n_splits
    target_size = sum(g["size"] for g in groups) / n_splits
    label_counts = np.zeros((n_splits, 2), dtype=float)
    sizes = np.zeros(n_splits, dtype=float)
    assignments: list[list[dict]] = [[] for _ in range(n_splits)]

    for group in groups:
        candidates = rng.permutation(n_splits)
        best_fold, best_score = None, None
        addition = np.array([group["n0"], group["n1"]], dtype=float)
        for fold in candidates:
            trial_labels = label_counts.copy()
            trial_sizes = sizes.copy()
            trial_labels[fold] += addition
            trial_sizes[fold] += group["size"]
            label_score = np.sum(((trial_labels - target) / np.maximum(target, 1.0)) ** 2)
            size_score = np.sum(((trial_sizes - target_size) / max(target_size, 1.0)) ** 2)
            score = float(label_score + 0.15 * size_score)
            if best_score is None or score < best_score:
                best_fold, best_score = int(fold), score
        assignments[best_fold].append(group)
        label_counts[best_fold] += addition
        sizes[best_fold] += group["size"]

    all_ids = set(df["Variant_ID"].astype(str))
    folds = []
    for fold_idx, assigned in enumerate(assignments):
        test_ids = [variant_id for group in assigned for variant_id in group["ids"]]
        train_ids = sorted(all_ids - set(test_ids))
        folds.append({
            "fold": fold_idx,
            "train_variant_ids": train_ids,
            "test_variant_ids": test_ids,
        })
    return folds


def validate_folds(df: pd.DataFrame, folds: list[dict], n_splits: int) -> dict:
    all_ids = set(df["Variant_ID"].astype(str))
    group_lookup = df.set_index("Variant_ID")["duplicate_group_id"].astype(str).to_dict()
    test_seen = []
    violations = []
    class_counts = []
    for fold in folds:
        train_ids, test_ids = set(fold["train_variant_ids"]), set(fold["test_variant_ids"])
        if train_ids & test_ids:
            violations.append(f"fold {fold['fold']}: train/test çakışması")
        if train_ids | test_ids != all_ids:
            violations.append(f"fold {fold['fold']}: eksik/fazla satır")
        test_seen.extend(test_ids)
        train_groups = {group_lookup[x] for x in train_ids}
        test_groups = {group_lookup[x] for x in test_ids}
        if train_groups & test_groups:
            violations.append(f"fold {fold['fold']}: duplicate group ayrıldı")
        block = df[df["Variant_ID"].astype(str).isin(test_ids)]
        counts = block["Label"].value_counts().to_dict()
        class_counts.append({"fold": fold["fold"], "n": len(block), "label0": int(counts.get(0, 0)), "label1": int(counts.get(1, 0))})
        if counts.get(0, 0) == 0 or counts.get(1, 0) == 0:
            violations.append(f"fold {fold['fold']}: bir sınıf eksik")
    if len(test_seen) != len(set(test_seen)) or set(test_seen) != all_ids:
        violations.append("test fold'ları tam bir partition oluşturmuyor")
    return {"ok": not violations, "violations": violations, "n_splits": n_splits, "fold_class_counts": class_counts}


def build_all(meta: pd.DataFrame | None = None) -> dict:
    meta = pd.read_csv(META_PATH) if meta is None else meta.copy()
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    validations = []
    for repeat in range(N_REPEATS):
        outer_seed = BASE_SEED + repeat
        outer = group_stratified_folds(meta, N_OUTER_SPLITS, outer_seed)
        outer_validation = validate_folds(meta, outer, N_OUTER_SPLITS)
        if not outer_validation["ok"]:
            raise AssertionError(outer_validation["violations"])
        validations.append({"repeat": repeat, **outer_validation})
        (SPLIT_DIR / f"outer_fold_repeat{repeat:02d}.json").write_text(
            json.dumps({
                "repeat": repeat, "seed": outer_seed, "n_splits": N_OUTER_SPLITS,
                "splitter": "custom deterministic greedy group-stratified",
                "folds": outer,
            }, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for outer_fold in outer:
            outer_train = meta[meta["Variant_ID"].isin(outer_fold["train_variant_ids"])].copy()
            inner_seed = outer_seed * 100 + outer_fold["fold"]
            inner = group_stratified_folds(outer_train, N_INNER_SPLITS, inner_seed)
            inner_validation = validate_folds(outer_train, inner, N_INNER_SPLITS)
            if not inner_validation["ok"]:
                raise AssertionError(inner_validation["violations"])
            (SPLIT_DIR / f"inner_fold_repeat{repeat:02d}_outer{outer_fold['fold']}.json").write_text(
                json.dumps({
                    "repeat": repeat, "outer_fold": outer_fold["fold"], "seed": inner_seed,
                    "n_splits": N_INNER_SPLITS,
                    "splitter": "custom deterministic greedy group-stratified",
                    "folds": [
                        {
                            "fold": f["fold"],
                            "train_variant_ids": f["train_variant_ids"],
                            "val_variant_ids": f["test_variant_ids"],
                        } for f in inner
                    ],
                }, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    manifest = {
        "design": "outer 5-fold x 10 repeats, inner 4-fold",
        "group_awareness": "duplicate_group_id aynı olan satırlar hiçbir fold sınırını geçmez",
        "base_seed": BASE_SEED,
        "splitter": "custom deterministic greedy group-stratified",
        "source_dataset": "data/processed/kanser/00_KANSER_TARGET_AND_METADATA.csv",
        "n_rows": int(len(meta)),
        "n_outer_files": N_REPEATS,
        "n_inner_files": N_REPEATS * N_OUTER_SPLITS,
        "immutable": True,
    }
    (SPLIT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    validation_summary = {
        "ok": all(x["ok"] for x in validations),
        "repeats": N_REPEATS,
        "outer_folds_per_repeat": N_OUTER_SPLITS,
        "inner_folds_per_outer": N_INNER_SPLITS,
        "group_split_violations": 0,
        "outer_fold_class_counts": [item for v in validations for item in v["fold_class_counts"]],
    }
    (SPLIT_DIR / "validation_summary.json").write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"manifest": manifest, "validation": validation_summary}


def main():
    print(json.dumps(build_all(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
