# CS4641 Group 5 Machine Learning Project

## Project Overview
This project aims to develop machine learning models for analyzing and predicting heart-related health outcomes. 
The project is structured into various directories for data preprocessing, analysis, and implementation of both supervised and unsupervised learning algorithms.

## Directory Structure
- **/Data/**: Contains raw and preprocessed datasets.
- **/DataAnalysis/**: Scripts and notebooks for exploratory data analysis.
- **/GitHub_Pages/**: Resources for the project's GitHub Pages site.
     - **GitHub_Pages/Images/**: Images used for the GitHub Pages site, such as plots, graphs, and code snapshots
- **/Preprocessing/**: Scripts for data cleaning and preprocessing.
     - **/Preprocessing/PCA.py**: Script implementing PCA for dimensionality reduction
     - **/Preprocessing/cleaning.py**: Script for cleaning missing values from the dataset
- **/SupervisedLearning/**: Implementation of supervised learning algorithms.
     - **/SupervisedLearning/randomforest.py**: Script implementing the random forest model using provided dataset
     - **/SupervisedLearning/randomforest_cleaned.ipynb**: Jupyter notebook containing code for random forest model and visualizations and quantitative metrics representing model output
     - **/SupervisedLearning/KNN_jupyter.ipynb**: Jupyter notebook containing code for implemented K-Nearest Neighbors model, visualizations and plotting, and quantitative metrics representing model output
     - **/SupervisedLearning/NeuralNetworkModified.ipynb**: Jupyter notebook containing code for implemented Neural Network model, visualizations and plotting, and quantitative metrics representing model output
- **/index.md**: GitHub Page detailing all progress of the project in the form of a report

## Getting Started
### Prerequisites
- Python 3.10

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sunnypark12/cs4641_group5.git
   cd cs4641_group5

### Usage
**Data Preprocessing**
Navigate to the Preprocessing directory and run the preprocessing scripts to clean and prepare the data.

**Data Analysis**
Explore the data using the notebooks and scripts in the DataAnalysis directory.

**Model Training**
Train and evaluate models using the scripts in the SupervisedLearning directory.
