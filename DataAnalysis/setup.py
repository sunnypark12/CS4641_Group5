import os
import numpy as np 
import pandas as pd 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",None)
from sklearn import preprocessing
import matplotlib 
matplotlib.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# Function to load dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Function to check for missing values
def check_missing_values(df):
    return df.isnull().sum()

# Function to encode categorical variables
def encode_categorical_variables(df):
    return pd.get_dummies(df)

# Function to scale numerical features
def scale_numerical_features(df, numerical_features):
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

def convert_string_columns(df):
    """
    Converts string data to the appropriate type in the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The processed DataFrame with string data converted.
    """
    # Convert string data to the appropriate type
    string_col = df.select_dtypes(include="object").columns
    df[string_col] = df[string_col].astype("string")
    
    return df

def get_categorical_columns(df):
    """
    Returns a list of categorical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    list: A list of categorical column names.
    """
    string_col = df.select_dtypes("string").columns.to_list()
    return string_col

def get_numerical_columns(df, exclude_col):
    """
    Returns a list of numerical columns in the DataFrame excluding specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    exclude_col (str): The column to exclude from the numerical columns.

    Returns:
    list: A list of numerical column names excluding the specified column.
    """
    string_col = get_categorical_columns(df)
    num_col = df.columns.to_list()
    for col in string_col:
        num_col.remove(col)
    if exclude_col in num_col:
        num_col.remove(exclude_col)
    return num_col

