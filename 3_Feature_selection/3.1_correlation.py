import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
import json
import argparse
import os

def main(input_csv, season_filter_json):
    # Load data
    df = pd.read_csv(input_csv, parse_dates=["datetime"])

    # Replace '#DIV/0!' values with NaN
    df.replace('#DIV/0!', np.nan, inplace=True)

    # Load the JSON file with the data filter
    with open(season_filter_json, "r") as f:
        raw_date_ranges = json.load(f)

    # Convert date strings to datetime objects
    date_ranges = {}
    for station, years in raw_date_ranges.items():
        date_ranges[station] = {}
        for year, species in years.items():
            date_ranges[station][int(year)] = {}
            for sp, (start_str, end_str) in species.items():
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
                date_ranges[station][int(year)][sp] = (start_date, end_date)

    species_list = ["Alnus", "Betula", "Poaceae", "Quercus"]

    for species in species_list:
        df_combined = pd.DataFrame()

        for station, years in date_ranges.items():
            for year, species_dates in years.items():
                if species not in species_dates:
                    continue

                start_date, end_date = species_dates[species]

                mask = (
                    (df["STA"] == station)
                    & (df["datetime"] >= start_date)
                    & (df["datetime"] <= end_date)
                    & (df["datetime"].dt.year == year)
                )
                df_filtered = df[mask].copy()
                relevant_columns = [col for col in df.columns if col not in ['STA', 'year', 'datetime']]
                df_filtered = df_filtered[relevant_columns]

                df_combined = pd.concat([df_combined, df_filtered], axis=0)

        df_combined = df_combined.dropna()
        column_order = [col for col in df.columns if col not in ['STA', 'year', 'datetime']]

        # --- Spearman ---
        corr_matrix = df_combined.corr(method="spearman").loc[column_order, column_order]
        corr_matrix.to_csv(f"Spearman_matrix_{species}.csv")
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, center=0)
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout(pad=3)
        plt.savefig(f"Spearman_correlation_matrix_{species}.png")
        plt.close()

        # --- Pearson ---
        corr_p_matrix = df_combined.corr(method="pearson").loc[column_order, column_order]
        corr_p_matrix.to_csv(f"Pearson_matrix_{species}.csv")
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_p_matrix, cmap="coolwarm", vmin=-1, vmax=1, center=0)
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout(pad=3)
        plt.savefig(f"Pearson_correlation_matrix_{species}.png")
        plt.close()

        # --- Mutual Information ---
        mi_matrix = pd.DataFrame(index=column_order, columns=column_order, dtype=float)
        for target in column_order:
            X = df_combined.drop(columns=[target])
            y = df_combined[target]
            mi = mutual_info_regression(X, y, discrete_features=False)
            for idx, feature in enumerate(X.columns):
                mi_matrix.loc[feature, target] = mi[idx]

        mi_matrix = mi_matrix.fillna(0.0)
        mi_matrix.to_csv(f"Mutual_Information_matrix_{species}.csv")
        plt.figure(figsize=(14, 12))
        sns.heatmap(mi_matrix.astype(float), cmap="coolwarm")
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout(pad=3)
        plt.savefig(f"Mutual_Information_matrix_{species}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute correlation matrices for pollen species")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--json", type=str, required=True, help="Path to JSON filter file")
    args = parser.parse_args()

    main(input_csv=args.csv, season_filter_json=args.json)

