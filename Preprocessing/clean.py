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
from sklearn.preprocessing import LabelEncoder

import setup as su
import explotaryDataAnalysis as eda

def main():
    file_path = './Data/heart.csv'
    
    # Load the dataset
    heart_df = su.load_dataset(file_path)

    # convert string data to appropriate type
    heart_df = su.convert_string_columns(heart_df)

    categorical_columns = su.get_categorical_columns(heart_df)  # Get categorical columns
    numerical_columns = su.get_numerical_columns(heart_df, None)  # Get numerical columns excluding "HeartDisease"

    print(categorical_columns)
    print(numerical_columns)

   
if __name__ == "__main__":
    main()
