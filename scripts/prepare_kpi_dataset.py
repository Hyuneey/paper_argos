import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert KPI Challenge train.csv into ARGOS value,label,index CSVs."
    )
    parser.add_argument(
        "--input_csv",
        default="datasets/dataset_1_KPI/Preliminary_dataset/train.csv",
        help="KPI train CSV with timestamp,value,label,KPI ID columns.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/datasets/kpi_preliminary",
        help="Output directory for one ARGOS CSV per KPI ID.",
    )
    parser.add_argument(
        "--max_series",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = repo_root / input_csv
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, usecols=["timestamp", "value", "label", "KPI ID"])
    written = 0
    for kpi_id, group in df.groupby("KPI ID", sort=True):
        output_df = group.sort_values("timestamp")[["value", "label"]].copy()
        output_df["index"] = range(len(output_df))
        output_path = output_dir / f"{kpi_id}.csv"
        output_df.to_csv(output_path, index=False)
        written += 1
        if args.max_series is not None and written >= args.max_series:
            break

    print(f"Wrote {written} ARGOS KPI CSV file(s) to {output_dir}")


if __name__ == "__main__":
    main()
