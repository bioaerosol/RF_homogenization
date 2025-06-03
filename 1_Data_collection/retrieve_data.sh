#!/bin/bash

location=("Basel")

year=2023


	# Define the Poleno ID (only one)
	poleno_id="poleno-17"  # Your single Poleno ID

	# Define the time range (only one)
	start_time="1672527600"  # Example: Start time in Unix timestamp
	end_time="1695333600"    # Example: End time in Unix timestamp

	# Define a unique output filename based on Poleno ID and start time
	output_filename="${location}_${year}_${poleno_id}_${start_time}_${end_time}.csv"

	# Run the Python script with parameters
	python retrieve_hourly.py "$poleno_id" "$start_time" "$end_time" "$output_filename"

echo "Finished processing $poleno_id from $start_time to $end_time. Data saved to $output_filename."

	# Define the list of Poleno IDs
	poleno_id=("poleno-27")  # Add your Poleno IDs here

	# Define the time range (only one)
	start_time="1695333600"  # Example: Start time in Unix timestamp
	end_time="1704063600"    # Example: End time in Unix timestamp

	# Define a unique output filename based on Poleno ID and start time
	output_filename="${location}_${year}_${poleno_id}_${start_time}_${end_time}.csv"

	# Run the Python script with parameters
	python retrieve_hourly.py "$poleno_id" "$start_time" "$end_time" "$output_filename"

echo "Finished processing $poleno_id from $start_time to $end_time. Data saved to $output_filename."

year=2024

	# Define the list of Poleno IDs
	poleno_id=("poleno-27")  # Add your Poleno IDs here

	# Define the time range (only one)
	start_time="1704063600"  # Example: Start time in Unix timestamp
	end_time="1730329200"    # Example: End time in Unix timestamp

	# Define a unique output filename based on Poleno ID and start time
	output_filename="${location}_${year}_${poleno_id}_${start_time}_${end_time}.csv"

	# Run the Python script with parameters
	python retrieve_hourly.py "$poleno_id" "$start_time" "$end_time" "$output_filename"

echo "Finished processing $poleno_id from $start_time to $end_time. Data saved to $output_filename."

	# Define the Poleno ID (only one)
	poleno_id="poleno-17"  # Your single Poleno ID

	# Define the time range (only one)
	start_time="1730329200"  # Example: Start time in Unix timestamp
	end_time="1735686000"    # Example: End time in Unix timestamp

	# Define a unique output filename based on Poleno ID and start time
	output_filename="${location}_${year}_${poleno_id}_${start_time}_${end_time}.csv"

	# Run the Python script with parameters
	python retrieve_hourly.py "$poleno_id" "$start_time" "$end_time" "$output_filename"

echo "Finished processing $poleno_id from $start_time to $end_time. Data saved to $output_filename."

