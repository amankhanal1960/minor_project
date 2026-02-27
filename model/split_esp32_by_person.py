"""
Create train/val/test CSV splits using person-level grouping.

Why: no person should appear in more than one split.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


CLASS_TO_LABEL = {
    "cough": 1,
    "non_cough": 0,
}


def parse_person_id(json_path: Path) -> str | None:
    """Read person_id from metadata JSON."""
    if not json_path.exists():
        return None

    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    for key in ("person_id", "speaker_id", "subject_id"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def build_manifest(dataset_root: Path, require_person_id: bool) -> pd.DataFrame:
    """Build a file manifest with wav_path, label, person_id."""
    rows = []

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue

        for wav_path in sorted(class_dir.rglob("*.wav")):
            json_path = wav_path.with_suffix(".json")
            person_id = parse_person_id(json_path)

            if not person_id and require_person_id:
                raise RuntimeError(f"Missing person_id in metadata: {json_path}")

            if not person_id:
                person_id = "unknown"

            rows.append(
                {
                    "wav_path": str(wav_path.resolve()),
                    "label": label,
                    "person_id": person_id,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No wav files found under: {dataset_root}")

    return df


def valid_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Ensure both classes exist and person ids do not overlap."""
    if train_df["label"].nunique() < 2:
        return False
    if val_df["label"].nunique() < 2:
        return False
    if test_df["label"].nunique() < 2:
        return False

    p_train = set(train_df["person_id"])
    p_val = set(val_df["person_id"])
    p_test = set(test_df["person_id"])

    if p_train & p_val:
        return False
    if p_train & p_test:
        return False
    if p_val & p_test:
        return False

    return True


def split_by_person(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
    max_tries: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Try multiple random seeds until a valid person-level split is found."""
    for step in range(max_tries):
        rs = seed + step

        # Split test first
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        train_val_idx, test_idx = next(gss_test.split(df, groups=df["person_id"]))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Split val from train_val
        rel_val = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=rs + 1000)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["person_id"]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        if valid_split(train_df, val_df, test_df):
            return train_df, val_df, test_df

    raise RuntimeError("Could not build a valid grouped split. Try a different seed or collect more people.")


def print_stats(name: str, split_df: pd.DataFrame) -> None:
    counts = split_df["label"].value_counts().to_dict()
    people = split_df["person_id"].nunique()
    print(f"{name}: files={len(split_df)} people={people} counts={counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create grouped train/val/test CSVs by person_id")
    parser.add_argument("--dataset", default="model/esp32_dataset", help="Dataset root containing cough/non_cough")
    parser.add_argument("--out-dir", default="model", help="Where to write CSV files")
    parser.add_argument("--prefix", default="esp32_5s_grouped", help="Output filename prefix")
    parser.add_argument("--test-size", type=float, default=0.15, help="Fraction for test split")
    parser.add_argument("--val-size", type=float, default=0.15, help="Fraction for val split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-tries", type=int, default=500, help="Max attempts to find valid grouped split")
    parser.add_argument(
        "--allow-missing-person-id",
        action="store_true",
        help="Allow clips without person_id (they are grouped under 'unknown')",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    require_person_id = not args.allow_missing_person_id
    df = build_manifest(dataset_root, require_person_id=require_person_id)

    print(f"Total files: {len(df)}")
    print(f"Total people: {df['person_id'].nunique()}")
    print(f"Overall class counts: {df['label'].value_counts().to_dict()}")

    train_df, val_df, test_df = split_by_person(
        df=df,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        max_tries=args.max_tries,
    )

    train_path = out_dir / f"{args.prefix}_train.csv"
    val_path = out_dir / f"{args.prefix}_val.csv"
    test_path = out_dir / f"{args.prefix}_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print_stats("Train", train_df)
    print_stats("Val", val_df)
    print_stats("Test", test_df)

    print(f"Saved: {train_path.resolve()}")
    print(f"Saved: {val_path.resolve()}")
    print(f"Saved: {test_path.resolve()}")


if __name__ == "__main__":
    main()
