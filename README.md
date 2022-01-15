# Holmusk Dataset Challenge

## Objective:
To analyze the clinical and financial data of patients hospitalized for a certain condition and find insights about the drivers of cost of care

## Raw data and Python Scripts:
bill_amount.csv: Contains the amount for each unique bill id\
bill_id.csv: Records each admission and their respective bill id(s)\
clinical_data.csv: Records the clinical data for each admission (medical histories, preop medications, symptoms experienced, lab results etc)\
demographics.csv: Contains patient data on race, gender and resident status

holmusk_assessment_data_prep.py: Data prep script to clean merge raw data\
holmusk_assessment_model_building.py: Model building script to train and evaluate model

## Method:
1. Perform data prep on the raw data to consolidate as one main dataset
2. Data exploration to identify outliers and conduct data preprocessing
3. Build a predictive model to predict total bill amount based on input variables
4. Evaluate model and derive insights about the drivers for total bill amount

## To run:
1. Ensure that raw data files and Python scripts are saved in the same location
2. Change the working directory (wd) in "holmusk_assessment_model_building.py" to the location of the raw data files and Python scripts
3. Run "holmusk_assessment_model_building.py"


