#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a daily recharge forcing CSV.")
    parser.add_argument("recharge_csv", type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.recharge_csv)
    if "date" not in df.columns or "Recharge" not in df.columns:
        raise ValueError("Recharge CSV must contain `date` and `Recharge` columns.")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date")
    if args.start_date or args.end_date:
        start = pd.Timestamp(args.start_date) if args.start_date else df["date"].min()
        end = pd.Timestamp(args.end_date) if args.end_date else df["date"].max()
        df = df[df["date"].between(start, end)]
    if df.empty:
        raise ValueError("No recharge records fall within the requested plotting period.")

    output = args.output or args.recharge_csv.with_name("recharge_forcing.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(df["date"], df["Recharge"], color="blue", alpha=0.5, width=1.5)
    ax.set_ylabel("Recharge (mm/day)")
    ax.set_title(
        f"Daily Recharge Forcing "
        f"({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d})"
    )
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Recharge plot saved to {output}")


if __name__ == "__main__":
    main()
