import subprocess
import itertools

# List of external validation stations and target species
stations = ['Basel']
targets = ['Alnus']

# Additional common arguments
csv_file = "./CH_LT_filtered.csv"
season_json = "./season_filter.json"
scaling_method = "standard"
target_multiplier = "1.5"

# Iterate over all combinations of stations and targets
for station, target in itertools.product(stations, targets):
    print(f"\n=== Running with pred_station={station}, target_column={target} ===\n")

    subprocess.run([
        "python", "4.2_RF.py",
        "--station", station,
        "--target", target,
        "--csv_file", csv_file,
        "--season_json", season_json,
        "--scaling_method", scaling_method,
        "--target_multiplier", target_multiplier
    ])
