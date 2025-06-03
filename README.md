# RF_homogenization
The goal of this toolkit is to provide merged timeseries of automatic (Poleno) and manual (Hirst) pollen concentrations. For the method development purposes we predict the Hirst time series based on the Poleno measurements, meteorological and air pollution data.The reverse path is also outlined.

# Description of the example dataset

• Target species: Alnus, Betula, Poaceae, and Quercus (Hirst)
• Feature space: poleno taxa, PM10, mean_wind_dir, mean_wind_vel, max_wind_vel, sum_prec,
mean_temp, max_temp, mean_RH, min_RH, max_RH, year
• Measurement stations: Basel, Buchs, Geneva, Locarno, Lugano, Luzern, Payerne, Vil-
nius, Zurich
• Years: 2022 (Vilnius only), 2023, 2024 (except Luzern)
• Dataset file: CH_LT_filtered.csv

# Remarks
• For Switzerland (CH.csv) only, a datafile with an extended input feature set is also avail-
able. It contains aggregated hourly meteorological data, PM2.5 concentrations and water
droplet counts from the Poleno
• Note the naming convention of marking automatic species with lower case and manual
once with upper case letters

# Workflow
An overview of the complete workflow from data collection to validation can be found here:
[workflow.pdf](https://github.com/user-attachments/files/20573556/workflow.pdf)

# Data collection 
Data collection is specific to 
# Season selection

# EDA and feature selection

# Training and prediction

# Manual
[Homogenization_toolkit_manual1.0.pdf](https://github.com/user-attachments/files/20573617/Homogenization_toolkit_manual1.0.pdf)
