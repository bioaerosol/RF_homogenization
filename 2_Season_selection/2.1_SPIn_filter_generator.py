import pandas as pd
import json

def compute_threshold_dates(**kwargs):
    # Get parameters from kwargs or use defaults
    csv_path = kwargs.get('csv_path', "input.csv")
    low_q = kwargs.get('low_quantile', 0.05)
    high_q = kwargs.get('high_quantile', 0.95)
    years = kwargs.get('years', [2023, 2024])
    output_path = kwargs.get('output_path', "seasons.json")

    # Load the dataset
    df = pd.read_csv(csv_path, parse_dates=['datetime'])

    # Filter by years
    df = df[df['datetime'].dt.year.isin(years)]

    # Species of interest
    species = ["Alnus", "Betula", "Poaceae", "Quercus"]

    # Initialize result dictionary
    thresholds_dict = {}

    # Process by station and year
    for (station, year), group in df.groupby(['STA', df['datetime'].dt.year]):
        station = str(station)
        year = int(year)

        group = group.sort_values('datetime')

        total_sums = group[species].sum()
        low_thresholds = total_sums * low_q
        high_thresholds = total_sums * high_q
        cumulative = group[species].cumsum()

        for sp in species:
            low_date = group.loc[cumulative[sp] >= low_thresholds[sp], 'datetime'].min()
            high_date = group.loc[cumulative[sp] >= high_thresholds[sp], 'datetime'].min()

            if pd.notna(low_date) and pd.notna(high_date):
                date_pair = [low_date.strftime('%Y-%m-%d'), high_date.strftime('%Y-%m-%d')]
                thresholds_dict.setdefault(station, {}).setdefault(str(year), {})[sp] = date_pair

    # Output as JSON
    with open(output_path, "w") as json_file:
        json.dump(thresholds_dict, json_file, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute pollen season thresholds.")
    parser.add_argument("--csv", type=str, default="CH_LT_filtered.csv", help="Path to input CSV file")
    parser.add_argument("--json", type=str, default="season_filter.json", help="Path to output JSON file")
    parser.add_argument("--low_quantile", type=float, default=0.05, help="Lower quantile threshold (e.g., 0.05)")
    parser.add_argument("--high_quantile", type=float, default=0.95, help="Upper quantile threshold (e.g., 0.95)")
    parser.add_argument("--years", type=int, nargs='+', default=[2023, 2024], help="List of years to include (e.g., 2022 2023)")

    args = parser.parse_args()

    compute_threshold_dates(
        csv_path=args.csv,
        output_path=args.json,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
        years=args.years
    )
