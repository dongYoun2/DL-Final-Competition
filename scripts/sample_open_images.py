import os
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Sample Open Images dataset")
    parser.add_argument("--sample-size", type=int, default=200_000, help="Number of images to sample (default: 200000)")

    args = parser.parse_args()
    sample_size = args.sample_size
    assert sample_size % 1000 == 0, "Sample size must be a multiple of 1000"

    data_dir = Path("data/Open_Images")
    data_dir.mkdir(parents=True, exist_ok=True)
    train_parquet_file = data_dir / "train_list.parquet"

    print("Reading train parquet file...")
    train_df = pd.read_parquet(train_parquet_file)
    print(f"Loaded {len(train_df):,} images")

    print(f"Sampling {sample_size:,} images...")
    sample_df = train_df.sample(n=sample_size, random_state=42)
    sample_df.drop(columns=["Labels"], inplace=True)

    sample_csv_file = data_dir / f"train_sampled_{sample_size//1000}k.csv"
    print(f"Saving to {sample_csv_file}...")
    sample_df.to_csv(sample_csv_file, index=False)

    # need this file for downloading the images
    image_list_file = data_dir / f"train_sampled_{sample_size//1000}k.txt"
    print(f"Writing image list to {image_list_file}...")
    with open(image_list_file, "w") as f:
        for image_id in sample_df["ImageID"]:
            f.write(f"train/{image_id}\n")

    print("Done!")


if __name__ == "__main__":
    main()
