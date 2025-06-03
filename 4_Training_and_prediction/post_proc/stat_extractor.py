import glob
import re
import csv

# Define a pattern to extract Target and Station.
# This regex looks for lines like: "Target: Alnus, Station: Payerne"
target_station_pattern = re.compile(r"Target:\s*([\w\s]+),\s*Station:\s*([\w\s]+)")

# Define a pattern to extract Validation R^2 Score.
# This regex looks for lines like: "Validation R^2 Score for Alnus: 0.8207660715806706"
# It assumes that the target name matches the one we captured earlier.
val_r2_pattern = re.compile(r"Validation R\^2 Score for [\w\s]+:\s*([\d\.]+)")

# List to hold the extracted data
results = []

# Process all txt files in the current directory (adjust the path or use recursive search if needed)
for filename in glob.glob("*.txt"):
    with open(filename, "r") as file:
        content = file.read()
        
        # Extract Target and Station using the regex.
        ts_match = target_station_pattern.search(content)
        if not ts_match:
            print(f"Could not parse Target/Station in {filename}")
            continue
        
        target = ts_match.group(1).strip()
        station = ts_match.group(2).strip()
        
        # Extract Validation R^2 Score.
        val_r2_match = val_r2_pattern.search(content)
        if val_r2_match:
            val_r2 = val_r2_match.group(1).strip()
        else:
            print(f"Could not find Validation R^2 Score in {filename}")
            val_r2 = "N/A"
        
        results.append((target, station, val_r2))

# Write the results to a CSV file.
output_file = "validation_r2.csv"
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Target", "Station", "Validation R^2 Score"])
    writer.writerows(results)

print(f"Extraction complete. Results saved to {output_file}")

