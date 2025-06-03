import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import sklearn.utils
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from datetime import datetime
import argparse
import json

# Define any additional helper functions used in the script
def resample_quantiles(X, y, num_bins=10, target_multiplier=1.5):
    # Use qcut to create quantiles
    quantile_bins = pd.qcut(y, q=num_bins, labels=False, retbins=True, duplicates='drop')
    quantile_indices = quantile_bins[0].reset_index(drop=True)  # Reset index for alignment
    bins = quantile_bins[1]  # Bin edges

    # Reset index of y to match the quantile_indices
    y = y.reset_index(drop=True)

    # Print the bins for inspection
    print("Quantile bins:", bins)

    # Count the number of values in each bin
    bin_counts = quantile_indices.value_counts()

    # Calculate target number of total samples (e.g., double the dataset)
    target_total_samples = int(target_multiplier * len(X))

    # Allocate samples to bins proportionally
    bin_weights = bin_counts / bin_counts.sum()  # Get proportion of each bin
    bin_targets = (bin_weights * target_total_samples).astype(int)  # Scale by target size

    # Ensure at least one sample per bin
    bin_targets = bin_targets.clip(lower=1)

    # Upsample each quantile
    X_resampled = []
    y_resampled = []

    for i in range(num_bins):
        X_bin = X[quantile_indices == i]
        y_bin = y[quantile_indices == i]

        if len(X_bin) > 0:
            sample_size = max(bin_targets[i], len(X_bin))
            X_resampled.append(X_bin.sample(n=sample_size, replace=True, random_state=42))
            y_resampled.append(y_bin.sample(n=sample_size, replace=True, random_state=42))

    # Concatenate final dataset
    X_resampled = pd.concat(X_resampled, axis=0).reset_index(drop=True)
    y_resampled = pd.concat(y_resampled, axis=0).reset_index(drop=True)

    # Print the lengths of the original and resampled target variables
    print(f"Original target length: {len(y)}, Resampled target length: {len(y_resampled)}")
    # Ensure feature names are retained
    X_resampled.columns = X.columns
    return X_resampled, y_resampled

def plot_regression_results(y_true, y_pred, target, pred_station):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_facecolor("#f0f0f0")
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7,color='blue',s=140,zorder=3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--k')
    plt.grid(True, linewidth=0.6, color='white',zorder=0)
    r2 = r2_score(y_true, y_pred)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Actual Values",fontsize=12)
    plt.ylabel("Predicted Values",fontsize=12)
    plt.title(f"Regression Plot for {target}\nR² = {r2:.3f}")
    plt.tight_layout()
    plt.savefig(f"RF_{target}_regression_trainCHLT_val.png", dpi=300)
    plt.show()
def plot_pca_real_vs_synthetic(X_real, X_synthetic, target, pred_station, title="PCA: Real vs. Synthetic Data"):
    """
    Performs PCA and plots original vs. synthetic data.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np.vstack((X_real, X_synthetic)))

    # Split PCA-transformed data
    X_pca_real = X_pca[:len(X_real)]
    X_pca_synthetic = X_pca[len(X_real):]

    # Plot PCA results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca_real[:, 0], y=X_pca_real[:, 1], label="Original Data", alpha=0.6)
    sns.scatterplot(x=X_pca_synthetic[:, 0], y=X_pca_synthetic[:, 1], label="Synthetic Data", alpha=0.6)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.legend()
    plt.savefig(f'RF_{target}_PCA_trainCHLT_val.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(importance_matrix,target, pred_station):
    plt.figure(figsize=(8,5))
    ax = plt.gca()
    ax.set_facecolor("#f0f0f0")
    sorted_importance = importance_matrix[target].sort_values(ascending=False)
    sns.barplot(x=sorted_importance.index, y=sorted_importance.values, ax=ax, palette="Blues_r", width=0.8, zorder=3)
    plt.grid(True, linewidth=0.6, color='white',zorder=0)
    plt.title(f"{target}", fontsize=14)
    plt.xlabel("Feature", fontsize=14)
    plt.ylabel("Importance Score",fontsize=14)
    plt.xticks(sorted_importance.index, rotation=90,fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'RF_{target}_barplots_trainCHLT_val.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_residuals(y_true, y_pred, target,pred_station):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 6))
    plt.stem(y_pred, residuals, linefmt="b-", markerfmt="bo", basefmt="r-")  # Blue lines, red baseline
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)  # Zero reference line
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot for {target}")
    plt.savefig(f"RF_residual_plot_{target}_trainCHLT_val.png", dpi=300)
    plt.show()


def plot_time_series(y_true, y_pred, dates, target, station_column, filtered_data, data_range, pred_station):
    # Convert dates to datetime format if not already
    dates = pd.to_datetime(dates)
    # Get unique stations
    station_names = {pred_station}

    print(station_names)
    for station in station_names:
        station_mask = station_column == station

        for year in [2023, 2024]:
            mask = station_mask & (dates.dt.year == year)

            print(f"Station: {station} | Year: {year} | Data count: {mask.sum()}")  # Debug

            if mask.sum() == 0:
                continue  # Skip if no data for this station and year

            fig, ax = plt.subplots(figsize=(10, 5))  # Create a new figure for each station-year

            filtered_dates = pd.to_datetime(dates[mask], errors='coerce')
            x_vals = dates[mask].reset_index(drop=True)
            y_vals = y_true[mask].reset_index(drop=True)
            # Line plots for actual and predicted values
            sns.lineplot(x=x_vals, y=y_vals, ax=ax, label='Measured Hirst', linestyle=':', linewidth=2, color='blue')
            sns.lineplot(x=dates[mask], y=y_pred[mask], ax=ax, label='Predicted Hirst', linestyle=':', linewidth=2, color='orange')
            sns.lineplot(x=dates[mask], y=filtered_data.loc[mask, target.lower()], ax=ax, label='Measured Poleno', linestyle=':', linewidth=2, color='green')
            ax.set_facecolor("#f0f0f0")
            ax.grid(True, linestyle='-', alpha=1, linewidth=0.6, color='white')
            ax.set_title(f"{station} - {year}")
            ax.set_xlabel("Date", fontsize=14)
            ax.set_ylabel(target,fontsize=14)
            ax.tick_params(axis="x", rotation=45, labelsize=14)
            ax.tick_params(axis="y", labelsize=14)

            plt.tight_layout()
            # Save figure for each station and year
            filename = f"RF_{target}_timeseries_trainCHLT_val_{pred_station}_{year}.png"
            plt.savefig(filename, dpi=300)
            plt.show  # Close the figure to free memory
            plt.close(fig)
def compute_feature_importance(data, date_ranges,
                                scaling_method='minmax', station=None, target=None,
                                target_multiplier=1.5, years=[2023, 2024]):
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.replace('#DIV/0!', np.nan, inplace=True)
    data = data.dropna()
    years = years

    pred_station = station
    target_columns = [target]
    stations = list(set(data['STA']) - {pred_station})

    exclude_columns = ['Poaceae','Quercus', 'Betula', 'Alnus', 'hmean_temp', 'PM2.5',
                       'hmean_pres', 'hmax_temp', 'hmean_wind_vel', 'hmean_wind_dir',
                       'hsum_prec', 'hmean_RH', 'max_temp', 'min_temp', 'hmax_wind_vel', 'max_RH', 'min_RH', 'max_wind_vel']
    feature_columns = [col for col in data.columns if col not in exclude_columns + ['STA','datetime', 'year']]
    importance_matrix = pd.DataFrame()

    scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
    imputer = SimpleImputer(strategy="median")

    for target in target_columns:
        train = pd.DataFrame()
        val = pd.DataFrame()
        for year in years:
            for station in stations:
                if year not in date_ranges[station] or target not in date_ranges[station][year]:
                    continue
                start, end = date_ranges[station][year][target]
                train_part = data[(data['datetime'] >= start) & (data['datetime'] <= end) & (data["STA"] != pred_station)]
                val_part = data[(data['datetime'] >= start) & (data['datetime'] <= end) & (data["STA"] == pred_station)]
                train = pd.concat([train, train_part])
                val = pd.concat([val, val_part])
        train.drop_duplicates(inplace=True)
        val.drop_duplicates(inplace=True)
        if train.empty or val.empty:
            print(f"No data available for target {target} in the specified date range.")
            continue

        print("This is the training set")
        print(train["STA"].unique())

        X = train[feature_columns]
        y = train[target]
        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

        if len(y) < 10:
            continue

        if len(y) < 20000:
            X_resampled, y_resampled = resample_quantiles(
                pd.DataFrame(X_scaled, columns=feature_columns), y, num_bins=10, target_multiplier=target_multiplier)
            print(f"Applied quantile resampling for {target}: {len(y)} → {len(y_resampled)} samples")
        else:
            X_resampled, y_resampled = X_scaled, y
            X_resampled = pd.DataFrame(X_resampled, columns=feature_columns)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        if val.empty:
            print(f"No data available for target {target} in the specified date range.")
            continue

        print("This is the validation station:", val["STA"].unique())

        X_val = val[feature_columns]
        y_val = val[target]
        X_val = imputer.transform(X_val)
        X_val = scaler.transform(X_val)
        X_val = pd.DataFrame(X_val, columns=feature_columns)

        param_grid = {
            'n_estimators': [50, 100, 500],
            'max_depth': [None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 10, 15],
            'max_features': ['sqrt'],
            'bootstrap': [False]
        }

        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        print("Best parameters:", grid_search.best_params_)
        print("Best R² score:", grid_search.best_score_)

        best_rf = RandomForestRegressor(criterion='squared_error', **grid_search.best_params_, random_state=42)
        cross_val_r2 = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='r2')
        print(f"Cross-validated R² scores for {target}: {cross_val_r2}")
        print(f"Mean R²: {cross_val_r2.mean():.3f}, Std: {cross_val_r2.std():.3f}")
        best_rf.fit(X_train, y_train)

        target_importance = pd.Series(best_rf.feature_importances_, index=feature_columns)
        top_features = target_importance.nlargest(23).index
        print(top_features)

        y_pred = best_rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE for {target}: {mse:.3f}")
        print(f"Test R^2 Score for {target}: {r2_score(y_test, y_pred)}")

        plot_regression_results(y_test, y_pred, target, pred_station)
        importance_matrix[target] = target_importance[top_features]
        plot_residuals(y_test, y_pred, target, pred_station)
        plot_feature_importance(importance_matrix, target, pred_station)

        y_pred2 = best_rf.predict(X_val)
        mse_val = mean_squared_error(y_val, y_pred2)
        print(f"Validation MSE for {target}: {mse_val:.3f}")
        print(f"Validation R^2 Score for {target}: {r2_score(y_val, y_pred2)}")

        with open(f"crossval_scores_{target}_{pred_station}.txt", "a") as f:
            f.write(f"Target: {target}, Station: {pred_station}\n")
            f.write(f"Cross-validated R² scores: {cross_val_r2}\n")
            f.write(f"Top Features:\n{top_features.to_list()}\n")
            f.write(f"Mean R²: {cross_val_r2.mean():.3f}, Std: {cross_val_r2.std():.3f}\n")
            f.write(f"Test MSE for {target}: {mse:.3f}\n")
            f.write(f"Test R^2 Score for {target}: {r2_score(y_test, y_pred)}\n")
            f.write(f"Validation MSE for {target}: {mse_val:.3f}\n")
            f.write(f"Validation R^2 Score for {target}: {r2_score(y_val, y_pred2)}\n")
            f.write("-" * 40 + "\n")

        plot_time_series(y_val, y_pred2, val['datetime'], target, val['STA'], val, date_ranges, pred_station)

        csv_filename = f"RF_{target}_predictions_trainCHLT_val_{pred_station}.csv"
        df_out = pd.DataFrame({'datetime': val['datetime'], 'predicted': y_pred2})
        df_out.to_csv(csv_filename, index=False)

    return importance_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Validate Random Forest Model.')
    parser.add_argument('--station', required=True, help='Validation station name')
    parser.add_argument('--target', required=True, help='Target variable (species)')
    parser.add_argument('--scaling_method', choices=['standard', 'minmax'], default='minmax', help='Scaling method')
    parser.add_argument('--csv_file', required=True, help='Input CSV file with environmental and target data')
    parser.add_argument('--season_json', required=True, help='JSON file with season date ranges')
    parser.add_argument('--target_multiplier', type=float, default=1.5, help='Target multiplier for resampling')
    parser.add_argument('--years', type=int, nargs='+', default=[2023, 2024],
                        help='List of years to include in training/validation')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    with open(args.season_json, "r") as f:
        raw_date_ranges = json.load(f)

    date_ranges = {}
    for station, years in raw_date_ranges.items():
        date_ranges[station] = {}
        for year, species in years.items():
            date_ranges[station][int(year)] = {}
            for sp, (start_str, end_str) in species.items():
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
                date_ranges[station][int(year)][sp] = (start_date, end_date)

    feature_importance = compute_feature_importance(
        df,
        date_ranges,
        scaling_method=args.scaling_method,
        station=args.station,
        target=args.target,
        target_multiplier=args.target_multiplier,
        years=args.years
    )

    print(feature_importance)
