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



# Print the dataframe description
# df.describe().T

# print(df.describe())
# print(df.head())

# # correlation matrix
# fig = px.imshow(df[num_col].corr(),title="Correlation Plot of the Heat Failure Prediction")
# fig.show()
# Compute correlation matrix only for numerical columns
corr_matrix = df[num_col].corr()

# Plot the correlation matrix using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Plot of the Heart Failure Prediction")
# plt.show()

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

# draw_histogram(df, x_column="HeartDisease", color_column="Sex", title="Distribution of Heart Diseases")

# plt.figure(figsize=(15,10))
# sns.pairplot(df,hue="HeartDisease")
# plt.title("Looking for Insites in Data")
# plt.legend("HeartDisease")
# plt.tight_layout()
# plt.plot()
# plt.show()

# plt.figure(figsize=(15,10))
# for i,col in enumerate(df.columns,1):
#     plt.subplot(4,3,i)
#     plt.title(f"Distribution of {col} Data")
#     sns.histplot(df[col],kde=True)
#     plt.tight_layout()
#     plt.plot()
#     plt.show()  # Show the plot

fig = px.box(df,y="Age",x="HeartDisease",title=f"Distrubution of Age")
fig.show()
