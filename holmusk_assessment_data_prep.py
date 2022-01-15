
#########################################
###### Importing required packages ######
#########################################

import pandas as pd
import numpy as np

###############################################################################################################################################################################

def total_bill_per_admission(bill_id_df, bill_amount_df):
    
    '''
    Using the raw bill_id and bill_amount files to calculate the total bill amount per admission
    Note that each patient id can have more than one admissions
    Identify each unique admission using patient id & admission date
    '''
    
    patient_id = "patient_id"
    bill_id = "bill_id"
    admission_date = "date_of_admission"
    total_bill_amount = "total_bill_amount"
    
    bill_per_admission_df = pd.merge(bill_id_df, bill_amount_df, how = "left", left_on = bill_id, right_on = bill_id)
    bill_per_admission_df = pd.DataFrame(bill_per_admission_df.groupby(by = [patient_id, admission_date]).sum())
    bill_per_admission_df = bill_per_admission_df.rename(columns = {"amount": total_bill_amount})
    bill_per_admission_df = bill_per_admission_df.reset_index()
    bill_per_admission_df = bill_per_admission_df[[patient_id, admission_date, total_bill_amount]]
    
    return bill_per_admission_df


def clean_categorical_field(df, field, keyword, value):
    
    '''
    Helper function to process categorical field in dataframe
    For each value in the "field" argument, if the value starts with the "keyword" argument, it is replaced with the "value" argument
    '''
    
    df[field] = df[field].str.upper()
    df[field] = [value if row_value.startswith(keyword) else row_value for row_value in df[field]]
    
    return df


def encoding_categorical_field(df, field):
    
    '''
    Helper function to encode categorical fields
    Uses the highest frequency value of the categorical field as the benchmark
    '''
    mode = df[field].mode()[0]
    field_unique = list(df[field].unique())
    field_unique.remove(mode)
    
    for value in field_unique:
        df[field + "_" + value]  = [1 if row_value == value else 0 for row_value in df[field]]
    
    del df[field]
            
    return df


def demographics_data_clean(demographics_df):
    
    '''
    Cleaning the raw demographics data by standardizing values in categorical fields (gender, race, resident_status),
    encoding the categorical fields using the highest frequency value as the benchmark,
    and also coverting date_of_birth to datetime
    '''
    
    gender = "gender"
    race = "race"
    resident_status = "resident_status"
    date_of_birth = "date_of_birth"
    
    demographics_clean_df = demographics_df.copy(deep = True)
    
    demographics_clean_df = clean_categorical_field(demographics_clean_df, gender, "F", "FEMALE")
    demographics_clean_df = clean_categorical_field(demographics_clean_df, gender, "M", "MALE")
    demographics_clean_df = clean_categorical_field(demographics_clean_df, race, "INDIA", "INDIAN")
    demographics_clean_df = clean_categorical_field(demographics_clean_df, resident_status, "SINGAPORE", "SINGAPORE CITIZEN")
    
    demographics_clean_df = encoding_categorical_field(demographics_clean_df, gender)
    demographics_clean_df = encoding_categorical_field(demographics_clean_df, race)
    demographics_clean_df = encoding_categorical_field(demographics_clean_df, resident_status)

    demographics_clean_df[date_of_birth] = pd.to_datetime(demographics_clean_df[date_of_birth])

    return demographics_clean_df


def clinical_data_clean(clinical_df):
    
    '''
    Cleaning the raw clinical data by dropping rows with empty columns (medical_history_2, medical_history_5), 
    standardizing values in medical_history_3 ("N": 0, "Y": 1),
    and renaming "id" as "patient_id"
    '''
    
    patient_id = "patient_id"
    medical_history_2 = "medical_history_2"
    medical_history_3 = "medical_history_3"
    medical_history_5= "medical_history_5"

    clinical_df = clinical_df.dropna(subset = [medical_history_2], axis = 0)
    
    clinical_df[medical_history_3] = clinical_df[medical_history_3].astype(str)
    clinical_df = clean_categorical_field(clinical_df, medical_history_3, "N", "0")
    clinical_df = clean_categorical_field(clinical_df, medical_history_3, "Y", "1")
    clinical_df[medical_history_3] = clinical_df[medical_history_3].astype(int)
    
    clinical_df = clinical_df.dropna(subset = [medical_history_5], axis = 0)

    clinical_df = clinical_df.rename(columns = {"id": patient_id})
    
    return clinical_df
    

def main_df_prep(bill_per_admission_df, demographics_df, clinical_df):
    
    '''
    Merging the cleaned bill per admission, demographics and clinical data frames into a single main data frame,
    converting date_of_admission and date_of_discharge into datetime,
    and calculating age_upon_admission and length_of_stay_days
    '''
    
    patient_id = "patient_id"
    date_of_admission = "date_of_admission"
    date_of_discharge = "date_of_discharge"
    date_of_birth = "date_of_birth"
    
    age_upon_admission = "age_upon_admission"
    length_of_stay_days = "length_of_stay_days"
    
    main_df = pd.merge(clinical_df, demographics_df, how = "left", left_on = patient_id, right_on = patient_id)    
    main_df = pd.merge(main_df, bill_per_admission_df, how = "left", left_on = [patient_id, date_of_admission], right_on = [patient_id, date_of_admission])
    
    main_df[date_of_admission] = pd.to_datetime(main_df[date_of_admission])
    main_df[date_of_discharge] = pd.to_datetime(main_df[date_of_discharge])
    
    main_df[age_upon_admission] = np.floor((main_df[date_of_admission] - main_df[date_of_birth]).astype('timedelta64[D]')/365)
    main_df[length_of_stay_days] = (main_df[date_of_discharge] - main_df[date_of_admission]).astype('timedelta64[D]')
    
    main_df = main_df.drop([patient_id, date_of_admission, date_of_discharge, date_of_birth], axis = 1)
    
    return main_df


def data_prep(bill_id_raw_df, bill_amount_raw_df, demographics_raw_df, clinical_raw_df):
    
    '''
    Main data prep function that takes in all raw data files as arguments and outputs a single main data frame
    '''

    total_bill_per_admission_clean_df = total_bill_per_admission(bill_id_raw_df, bill_amount_raw_df)
    
    demographics_clean_df = demographics_data_clean(demographics_raw_df)
    
    clinical_clean_df = clinical_data_clean(clinical_raw_df)
    
    main_df = main_df_prep(total_bill_per_admission_clean_df, demographics_clean_df, clinical_clean_df)
    
    return main_df
    