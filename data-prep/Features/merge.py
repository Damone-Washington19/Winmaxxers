from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
YEARS = list(range(2020, 2027))


def find_yearly_files():
    files = []
    for y in YEARS:
        p = SCRIPT_DIR / f"{y}_features.csv"
        if p.exists():
            files.append((y, p))
    return files


def merge_yearly_features(out_name="all_years_features.csv"):
    files = find_yearly_files()
    if not files:
        raise FileNotFoundError("No per-year feature files found in Features/ folder")

    parts = []
    for year, path in files:
        df = pd.read_csv(path, low_memory=False)
        # ensure year column exists and is correct
        if "year" not in df.columns:
            df["year"] = year
        else:
            df["year"] = df["year"].fillna(year).astype(int)

        parts.append(df)

    merged = pd.concat(parts, ignore_index=True)

    # drop exact duplicates (same title+year), keep first occurrence
    if "title" in merged.columns and "year" in merged.columns:
        merged = merged.drop_duplicates(subset=["title", "year"], keep="first")

    out_path = SCRIPT_DIR / out_name
    merged.to_csv(out_path, index=False)

    print(f"Merged {len(parts)} files -> {out_path} ({len(merged)} rows)")
    return out_path


if __name__ == "__main__":
    merge_yearly_features()
