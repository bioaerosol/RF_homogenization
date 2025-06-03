import requests
import base64
import time
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import sys

api_host = "url to data explore"
username = "" # The username you use for the DataExplorer
password = "" # The password you use for the DataExplorer
classifier_config_id = "11ee982c-cb32-88c1-bd0d-ae7b87f820b4" # The classifier config to use (check on DataExplorer)

poleno_id = sys.argv[1]
start_time = sys.argv[2]
end_time = sys.argv[3]
output_filename = sys.argv[4]
# Function to convert human readable time (UTC) to UNIX timestamp
def convert_to_unix(time_str):
    return int(datetime.strptime(time_str, "%Y-%m-%d %H:%M").timestamp())

# Define from when to when yo§u would like to compute the concentrations
from_ = convert_to_unix(start_time) #int(datetime(2023, 4, 30, 23, 00).timestamp())
to_ = convert_to_unix(end_time) #int(datetime(2023, 5, 10, 23, 00).timestamp())

# Initialize the start time and end time
current_time = from_
print(current_time)
# Dictionary to store concentrations for each hour
data_by_hour = {}

# Start timing the download
st_time = time.time()

# Loop through the hours between the from_ and to_ dates
while current_time < to_:
    next_hour = current_time + 3600  # 3600 seconds = 1 hour

    # Build the URL for this specific hour
    url = f"{api_host}/api/v1/devices/{poleno_id}/classifiers/{classifier_config_id}/concentrations?from={current_time}&to={next_hour}&lowthreshold=0&supervisordisable=true"

    # Create the auth token
    auth_str = f"{username}:{password}"

    # Make the request
    req = requests.get(
        url=url,
        headers={
            "Authorization": f"Basic {base64.b64encode(auth_str.encode()).decode()}"
        }
    )

    # Check if the request was a success
    if req.status_code != 200:
        print(f"There was an issue with your query for hour starting {current_time}.")
        print("Status code: ", req.status_code)
        current_time = next_hour  # Skip this hour and continue
        continue

    # Parse the response
    response = req.json()
    con_json = response["concentrations"]

    # Collect data for the current hour
    time_key = datetime.utcfromtimestamp(next_hour).strftime("%Y-%m-%d %H:%M:%S")
    data_by_hour[time_key] = {class_name: data["concentration"] for class_name, data in con_json.items()}

    # Move to the next hour
    current_time = next_hour
    print(current_time)
en_time = time.time()
elapsed_time=en_time-st_time
print(f"Download completed in {elapsed_time:.2f} seconds.")
# Prepare the reshaped CSV data
# Get all unique class names as headers
all_classes = set()
for hour_data in data_by_hour.values():
    all_classes.update(hour_data.keys())

# Sort class names for consistency
all_classes = sorted(all_classes)

# Write the reshaped data to CSV
with open(output_filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write header: first column is time, followed by class names
    writer.writerow(["Time"] + all_classes)

    # Write rows: each row corresponds to a time, with the average concentrations for each class
    for time_key, concentrations in data_by_hour.items():
        row = [time_key] + [concentrations.get(class_name, 0) for class_name in all_classes]
        writer.writerow(row)

print(f"Reshaped data successfully written to {output_filename}")

