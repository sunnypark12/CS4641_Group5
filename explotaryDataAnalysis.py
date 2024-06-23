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

def create_correlation_matrix(df, columns, column_type='numerical', title="Correlation Matrix"):
    """
    Creates a correlation matrix using Seaborn and computes correlation for either numerical or categorical columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): The columns to include in the correlation computation.
    column_type (str): The type of columns to compute correlation for ('numerical' or 'categorical').
    title (str): The title of the correlation matrix plot.

    Returns:
    None
    """
    if column_type == 'numerical':
        # Compute correlation matrix for numerical columns
        corr_matrix = df[columns].corr()
    elif column_type == 'categorical':
        # Convert categorical columns to numerical using Label Encoding
        df_encoded = df[columns].apply(lambda col: pd.factorize(col, sort=True)[0])
        corr_matrix = df_encoded.corr()
    else:
        raise ValueError("column_type must be either 'numerical' or 'categorical'")

    # Plot the correlation matrix using seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.show()

def draw_histogram(df, x_column, color_column, title):
    """
    Draws a histogram using Plotly Express.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    x_column (str): The column to be used on the x-axis.
    color_column (str): The column to be used for color grouping.
    title (str): The title of the histogram.

    Returns:
    None
    """
    fig = px.histogram(
        df,
        x=x_column,
        color=color_column,
        hover_data=df.columns,
        title=title,
        barmode="group"
    )
    fig.show()

def create_pairplot(df, hue_column=None, title="Pairplot"):
    """
    Creates a pairplot using Seaborn to show pairwise bivariate distributions in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    hue_column (str): The column to use for hue (optional).
    title (str): The title of the pairplot.

    Returns:
    None
    """
    plt.figure(figsize=(15, 10))
    sns.pairplot(df, hue=hue_column)
    plt.suptitle(title, y=1.02)  # Adjust y for the title to be above the plot
    plt.tight_layout()
    plt.show()

def plot_distributions(df, title="Distribution of Data"):
    """
    Plots the distribution of each variable in the DataFrame with a kernel density estimate.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    title (str): The overall title of the plots.

    Returns:
    None
    """
    num_columns = len(df.columns)
    num_rows = (num_columns // 3) + 1 if num_columns % 3 != 0 else num_columns // 3
    
    plt.figure(figsize=(15, num_rows * 5))
    
    for i, col in enumerate(df.columns, 1):
        plt.subplot(num_rows, 3, i)
        plt.title(f"Distribution of {col} Data")
        sns.histplot(df[col], kde=True)
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()