# RF_homogenization
The goal of this toolkit is to provide merged timeseries of automatic (Poleno) and manual (Hirst) pollen concentrations. For the method development purposes we predict the Hirst time series based on the Poleno measurements, meteorological and air pollution data.The reverse path is also outlined.

# Description of the example dataset

• Target species: Alnus, Betula, Poaceae, and Quercus (Hirst)

• Feature space: poleno taxa, PM10, mean_wind_dir, mean_wind_vel, max_wind_vel, sum_prec,
mean_temp, max_temp, mean_RH, min_RH, max_RH, year

• Measurement stations: Basel, Buchs, Geneva, Locarno, Lugano, Luzern, Payerne, Vilnius, Zurich

• Years: 2022 (Vilnius only), 2023, 2024 (except Luzern)

• The data folder contains the input file with all stations and features (CH_LT_filtered.csv) as well as a json file (season_filter.json) that contains station and species specific thresholds to select the relevant season for the EDA and the RF regression.

# Remarks
• For Switzerland (CH.csv) only, a datafile with an extended input feature set is also avail-
able. It contains aggregated hourly meteorological data, PM2.5 concentrations and water
droplet counts from the Poleno.

• Note the naming convention of marking automatic species with lower case and manual
once with upper case letters

# Workflow
An overview of the complete workflow from data collection to validation can be found here:
[workflow.pdf](https://github.com/user-attachments/files/20573556/workflow.pdf)
Detailed description is of each step is available in the pdf manual.
[Homogenization_toolkit_manual1.0.pdf](https://github.com/user-attachments/files/20573617/Homogenization_toolkit_manual1.0.pdf)

# Data collection 
Data collection is specific to the participating institutes and is not detailed here, the manual contains data sources. The repository only contains the scripts to download Poleno data.

# Season selection
2.1_SPIn_filter_generator.py  creates a json file with filters to select the pollen season for each station, species and year.
Usage:

python 2.1_SPIn_filter_generator.py --csv infile.csv --json season.json --low_quantile q1 --high_quantile q2 --years YYYY1 YYYY2

# EDA and feature selection
3.1_correlation.py generates Pearson and Spearman correlation matrices and mutual information matrices that are used to remove high correlated features
Usage:

python 3.1_correlation.py --csv infile.csv --json season.json

# Training and prediction
Training of the RF model and the prediction of the timeseries for the validation station (not included in the training set) is performed by running 4.1\_run\_RF.py.

Arguments:

•- -stations: the list of external validation stations (list, use "none" for including all stations
in the training dataset)

• - -targets: list of target species (list)

• - -csv_file: input data (string)

• - -season_json: season filter (string)

• - -scaling_method: scaling type (string, choices: ["standard","minmax"]

• - -target_multiplier: factor for data augmentation (float, default:1.5)

• –years: list of years to include in the training (list, defaults: ["2023", "2024"])

4.2\_RF.py performs the data augmentation, grid search hyperparameter tuning, the training, testing, 5-fold cross validation and external validation. Main parameters can be controlled from the run control script (4.1_run_RF.py), while the grid for the grid search as well as the stem name of the output files can be changed within the code. Features to exclude from the regression based on the results of the EDA can be modified by updating these lines in 4.2\_RF.py:

exclude_columns = [’Poaceae’,’Quercus’, ’Betula’, ’Alnus’, ’hmean_temp’, ’PM2.5’, ’hmean_pres’, ’hmax_temp’, ’hmean_wind_vel’, ’hmean_wind_dir’,’hsum_prec’, ’hmean_RH’, ’max_temp’, ’min_temp’, 'hmax_wind_vel’, ’max_RH’, ’min_RH’, ’max_wind_vel’]

Output:

• Text file containing the statistical scores of the cross validation, testing and external vali-
dation (txt),

• Barplots of importance of each feature for the random forest regression (png),

• Scatter plots of the first two PCs of the original dataset and the synthetic dataset from the
augmentation protocol (png),

• Regression plots of the test dataset (png),

• Timeseries of the Predicted data overlayed with the Automatic and Manual measurements
for all years separately for the external validation station, only produced if the validation
station is not set to "none" (png),

• Timeseries of the predicted data, only produced if the validation station is not set to "none"
(csv),

• The saved model, imputer and scaler (joblib).

# Additional tools

4_Training_and_prediction/post_proc contains additional python scripts that convert the output into doc files and collect and plot statistical measures

