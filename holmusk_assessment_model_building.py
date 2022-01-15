
#########################################
###### Importing required packages ######
#########################################

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import os
orig_wd = os.getcwd()
wd = r'C:\Users\angus\Documents\holmusk'
os.chdir(wd)
import holmusk_assessment_data_prep

#%%
################################################################
###### Reading in raw data files and performing data prep ######
################################################################

bill_id_raw_df = pd.read_csv("bill_id.csv")
bill_amount_raw_df = pd.read_csv("bill_amount.csv")
demographics_raw_df = pd.read_csv("demographics.csv")
clinical_raw_df = pd.read_csv("clinical_data.csv")

main_df = holmusk_assessment_data_prep.data_prep(bill_id_raw_df, bill_amount_raw_df, demographics_raw_df, clinical_raw_df)

#%%
###############################################################
###### Exploratory data analysis and data pre-processing ######
###############################################################

# Distritution Plot of Total Bill Amount
sns.distplot(main_df["total_bill_amount"])
plt.title("Distritution Plot of Total Bill Amount")

#%%
# Removing outliers for total_bill_amount. Outliers are defined as greater than 99th percentile
upper_q = main_df["total_bill_amount"].quantile(0.99)
main_df_without_outliers = main_df[main_df["total_bill_amount"] < upper_q]

#%%
# Distritution Plot of Total Bill Amount (without outliers)
sns.distplot(main_df_without_outliers["total_bill_amount"])
plt.title("Distritution Plot of Total Bill Amount (without outliers)")

# eplore_df is a data frame containing descriptive statistics for each variable
explore_df = main_df_without_outliers.describe(include = 'all')

# Finalizing pre-processed data
main_df_pre_preprocessed = main_df_without_outliers

# Listing all independent variables
independent_variables_dict = {}

independent_variables_dict["medical_history"] = ['medical_history_1', 'medical_history_2',
                                                 'medical_history_3', 'medical_history_4', 
                                                 'medical_history_5', 'medical_history_6',
                                                 'medical_history_7']

independent_variables_dict["preop_medication"] = ['preop_medication_1', 'preop_medication_2', 'preop_medication_3', 
                                                  'preop_medication_4', 'preop_medication_5', 'preop_medication_6']

independent_variables_dict["symptoms"] = ['symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5']

independent_variables_dict["lab_results"] = ['lab_result_1', 'lab_result_2', 'lab_result_3']

independent_variables_dict["demographics"] = ['gender_FEMALE', 
                                              'race_INDIAN', 'race_MALAY', 'race_OTHERS',
                                              'resident_status_PR', 'resident_status_FOREIGNER']

independent_variables_dict["others"] = ['weight', 'height', 'age_upon_admission', 'length_of_stay_days']

independent_variables = sum(independent_variables_dict.values(),[])

# Setting target and inputs
target = main_df_pre_preprocessed["total_bill_amount"]
inputs = main_df_pre_preprocessed[independent_variables]

# Scaling the inputs
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

# Splitting the data into test and training set
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, target, test_size = 0.2, random_state = 365)

#%%
###########################################
###### Model building and evaluation ######
###########################################

# Performing linear regression on the training set with total_bill_amount as the target variable
regression_1 = LinearRegression()
regression_1.fit(x_train, y_train)
y_hat = regression_1.predict(x_train)

#%%
# Plotting the predicted total bill amount against the actual bill amount
plt.scatter(y_train, y_hat)
plt.title("Predicted vs Actual total bill amount (training set)")
plt.xlabel("Actual total bill amount")
plt.ylabel("Predicted total bill amount")
# predicted total bill amount is slightly higher than actual total bill amount

#%%
# Plotting the distribution plot of the errors
sns.distplot(y_train - y_hat)
plt.title("Distritution Plot of Actual - Predicted total bill amount (training set)")
# They seem to be normally distributed, but with a longer left tail implies that the model tends to slightly overestimate the total bill amount

#%%
# Calculating the R Squared and Adjusted R Squared score
r_squared = regression_1.score(x_train, y_train) # R Squared score is 0.951, model explains about 95% of the variability
adj_r_sqaured = 1-((1-r_squared)*((len(y_train)-1)/(len(y_train)-len(independent_variables)-1))) # Adjusted R Squared score is 0.951

# Calculating the intercept
intercept = regression_1.intercept_ # Intercept value is 21454

# Finding the weights of each independent variable
coefficients = pd.DataFrame(data = independent_variables, columns = ["Features"])
coefficients["Weights"] = regression_1.coef_
coefficients_sorted = coefficients.sort_values(by = "Weights", ascending = False)

# Analyzing the weights based on category
coefficients_dict = {}
for category in list(independent_variables_dict.keys()):
    category_df = coefficients[coefficients["Features"].isin(independent_variables_dict[category])]
    if category == "demographics":
        category_df = category_df.sort_values(by = ["Features","Weights"], ascending = [True, False])
    else:
         category_df = category_df.sort_values(by = ["Weights"], ascending = False)       
    coefficients_dict[category] = category_df
    
coefficients_medical_history = coefficients_dict["medical_history"]
coefficients_preop_medication = coefficients_dict["preop_medication"]
coefficients_symptoms = coefficients_dict["symptoms"]
coefficients_lab_results = coefficients_dict["lab_results"]
coefficients_demographics = coefficients_dict["demographics"]
coefficients_others = coefficients_dict["others"]

# Continous variables
# A positive weight shows that as a feature increases in value, the predicted total bill amount increases
# A negative weight shows that as a feature increases in value, the predicted total bill amount decreases
# 
# Dummy variables
# A positive weight shows that the respective category is greater than the benchmark
# A negative weight shows that the respective category is lesser than the benchmark

#%%
#######################################
###### Testing model on test set ######
#######################################

# Testing linear regression model on test set
y_hat_test = regression_1.predict(x_test)

#%%
# Plotting the predicted total bill amount against the actual bill amount
plt.scatter(y_test, y_hat_test)
plt.title("Predicted vs Actual total bill amount (test set)")
plt.xlabel("Actual total bill amount")
plt.ylabel("Predicted total bill amount")

#%%
# Plotting the distribution plot of the errors
sns.distplot(y_test - y_hat_test)
plt.title("Distritution Plot of Actual - Predicted total bill amount (test set)")

#%%
# Analyzing the performance of the model (performance_df) on the test set
performance_df = pd.DataFrame(data = y_hat_test, columns = ["Prediction"])
performance_df["Actual"] = y_test.reset_index(drop = True)
performance_df["Residuals"] = performance_df["Prediction"] - performance_df["Actual"]
performance_df["Difference (%)"]  = (np.abs(performance_df["Residuals"]/ performance_df["Actual"])) * 100
performance_df = performance_df.sort_values(by = ["Difference (%)"], ascending = False)
performance_stats = performance_df.describe()

