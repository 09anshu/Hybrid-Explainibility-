import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_external_df(df: pd.DataFrame, image_root: str, image_col: str, label_col: str) -> pd.DataFrame:
    if image_col not in df.columns:
        raise ValueError(f"Missing image column: {image_col}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    out = pd.DataFrame()
    out["raw_path"] = df[image_col].astype(str)
    out["has_fracture"] = df[label_col].astype(float).clip(0, 1)

    if "patient_id" in df.columns:
        out["patient_id"] = df["patient_id"].astype(str)
    else:
        out["patient_id"] = out["raw_path"].apply(lambda p: Path(p).stem)

    if "source" in df.columns:
        out["source"] = df["source"].astype(str)
    else:
        out["source"] = "external"

    out["abs_path"] = out["raw_path"].apply(
        lambda p: p if os.path.isabs(p) else os.path.join(image_root, p)
    )

    out = out[out["abs_path"].apply(os.path.exists)].reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError("No existing images found after path resolution.")
    return out


def stratified_subset(df: pd.DataFrame, val_size: int, seed: int) -> pd.DataFrame:
    if val_size <= 0:
        raise ValueError("val_size must be > 0")

    if val_size >= len(df):
        return df.copy().reset_index(drop=True)

    strat_key = df["has_fracture"]
    if len(strat_key.unique()) < 2:
        # If external labels are single-class, return random sample.
        return df.sample(val_size, random_state=seed).reset_index(drop=True)

    _, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=strat_key,
    )
    return val_df.reset_index(drop=True)


def export_subset(df: pd.DataFrame, out_dir: str, link_mode: str = "symlink"):
    out_path = Path(out_dir)
    images_out = out_path / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in df.iterrows():
        src = Path(row["abs_path"])
        safe_name = f"{i:06d}_{src.name}"
        dst = images_out / safe_name

        if not dst.exists():
            if link_mode == "copy":
                data = src.read_bytes()
                dst.write_bytes(data)
            else:
                try:
                    os.symlink(src, dst)
                except FileExistsError:
                    pass
                except OSError:
                    # Symlink may be unsupported in some environments.
                    data = src.read_bytes()
                    dst.write_bytes(data)

        rows.append(
            {
                "subset_path": str(dst),
                "original_path": str(src),
                "has_fracture": int(row["has_fracture"]),
                "patient_id": row["patient_id"],
                "source": row["source"],
            }
        )

    subset_df = pd.DataFrame(rows)
    subset_csv = out_path / "fracture_validation_subset.csv"
    subset_df.to_csv(subset_csv, index=False)

    summary = {
        "num_samples": int(len(subset_df)),
        "num_positive": int((subset_df["has_fracture"] == 1).sum()),
        "num_negative": int((subset_df["has_fracture"] == 0).sum()),
        "positive_rate": float((subset_df["has_fracture"] == 1).mean()),
        "sources": subset_df["source"].value_counts().to_dict(),
    }
    with open(out_path / "subset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved subset CSV: {subset_csv}")
    print(f"Saved summary   : {out_path / 'subset_summary.json'}")
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build dedicated fracture validation subset from external labeled data"
    )
    parser.add_argument("--external-csv", required=True, help="Path to external labels CSV")
    parser.add_argument("--image-root", default=".", help="Root for relative image paths in external CSV")
    parser.add_argument("--image-col", default="image_path", help="Column containing image path")
    parser.add_argument("--label-col", default="has_fracture", help="Binary fracture label column")
    parser.add_argument("--val-size", type=int, default=500, help="Validation subset size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="fracture_validation_subset")
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ext_df = pd.read_csv(args.external_csv)
    norm_df = normalize_external_df(ext_df, args.image_root, args.image_col, args.label_col)
    val_df = stratified_subset(norm_df, args.val_size, args.seed)
    export_subset(val_df, args.out_dir, link_mode=args.link_mode)
