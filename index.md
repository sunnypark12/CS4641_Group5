# CS4641_Group5

**Group Members:** Joshua K Thomas, Patcharapong Aphiwetsa, SeungTaek(Stan) Lee, Shayahn Mirfendereski, Sunho(Sunny) Park

## Introduction/Background

According to the CDC [1], heart failure is the leading cause of death in the world, affecting over 600,000 people in the U.S. alone. There are more than a handful of machine learning models [2], [3], such as K Nearest Neighbor, Naive Bayes Classifier, and Ridge Classifier, that have been applied to predict heart failure by utilizing inputted clinical features. These features can include age, cholesterol levels, and blood pressure. The provided datasets contain comprehensive statistics of clinical factors relating to the risk of heart failure:

1. **[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) by fedesoriano:**
   - Features: Common event features caused by CVDs
2. **[Heart Failure Clinical Records](https://www.kaggle.com/datasets/nimapourmoradi/heart-failure-clinical-records) by Nima Pourmoradi:**
   - Features: Clinical records specific to heart failure
3. **[Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) by KAMIL PYTLAK:**
   - Features: Data collected from CDC, including all the major risk factors for heart failure

## Problem/Motivation

Despite significant advances in medical technology, accurately predicting heart failure remains a challenging task. Many existing models are not generalizable across different patient populations and often fail to provide actionable insights for healthcare providers. The complexity and variability of human physiology necessitate more sophisticated predictive models that can handle diverse and high-dimensional data. For instance, several research papers like [4] have identified various factors, including environmental, genetic, and lifestyle factors, that are correlated with heart failure risk, underscoring the need for improved predictive models that effectively capture and embody the relationships among them. Such models are crucial for identifying high-risk patients earlier and more accurately in a wider range of scope, enhancing treatment plans and reducing mortality rates. Leveraging machine learning models can provide robust, reliable predictions in clinical settings.

## Data Cleaning

Our group decided to utilize the first dataset we found from Kaggle, the **Heart Failure Prediction Dataset**. The dataset contained some missing values and inconsistencies that needed to be addressed before we could proceed with our analysis. Here, we describe the steps we took to clean the dataset and provide a comparison of the dataset before and after cleaning.

### Cleaning Process
**Handling Missing Values**:
   - We identified that the dataset had several missing values, particularly in the `RestingBP` and `Cholesterol` columns. These missing values were addressed by removing rows with zero values in these columns to ensure the accuracy and integrity of our data.

### Cleaning Code

```
python
import pandas as pd

# Load the dataset
file_path = './Data/heart.csv'
df = pd.read_csv(file_path)

# Remove rows where RestingBP or Cholesterol columns have zero values
cleaned_df = df[(df['RestingBP'] != 0) & (df['Cholesterol'] != 0)]

# Save the cleaned dataset to a new CSV file
cleaned_file_path = 'cleaned_heart.csv'
cleaned_df.to_csv(cleaned_file_path, index=False)

# Display the first few rows of the cleaned dataframe
print(cleaned_df.head())

print("Cleaned dataset saved to:", cleaned_file_path)
```

Below are the comparison images between the dataset before and after cleaning:

**Before Cleaning**
![Before_Cleaning](GitHub_Pages/Images/Before_Cleaning.png)

**After Cleaning**
![After_Cleaning](GitHub_Pages/Images/After_Cleaning.png)
As illustrated, the data cleaning process significantly improved the dataset quality, making it more suitable for subsequent analysis and modeling.

## Methods

1. **Supervised Methods:**
   - **KNN (scikit-learn):** For simple predictions based on similarity and visualization of the data.
   - **Random Forest Classification (sklearn):** For accurate predictions. In a study trying to predict type-2 diabetes, Random Forest performed the best out of 7 ML methods surveyed. Assuming that our disease prediction has a similar structure, using this technique would fit well.
   - **Neural Networks (pytorch):** For handling high-dimensional data with deep learning techniques. NN usually has the highest flexibility and thus can result in the best result for our model.

2. **Data Preprocessing:**
   - **Dimensionality Reduction with PCA:** To simplify data and remove redundant features and improve model performance.
   - **Fill in missing data (pandas):** To ensure a complete dataset.
   - **Joining different datasets (pandas):** To ensure data integration.

## Results/Discussion

1. **Validation Metrics:** Recall, Precision, F1-score, Conditional Entropy, Mutual Information.
2. **Project Goal:** Create an ML model that more accurately predicts the risk of heart disease, utilizing various datasets and algorithms to provide reliable insights for healthcare providers.
3. **Expected Results:** There exists a correlation between the risk of heart failure and external factors such as lifestyle, sleep, and genetics, with changes in these factors influencing said risk.

## References

[1] “FastStats,” Leading Causes of Death. https://www.cdc.gov/nchs/fastats/leading-causes-of-death.htm

[2] J. Wang, “Heart Failure Prediction with Machine Learning: A Comparative Study,” Journal of Physics. Conference Series, vol. 2031, no. 1, p. 012068, Sep. 2021, doi: 10.1088/1742-6596/2031/1/012068.

[3] M. Badawy, N. Ramadan, and H. A. Hefny, “Healthcare predictive analytics using machine learning and deep learning techniques: a survey,” Journal of Electrical Systems and Information Technology, vol. 10, no. 1, Aug. 2023, doi: 10.1186/s43067-023-00108-y.

[4] V. Escolar et al., “Impact of environmental factors on heart failure decompensations,” ESC Heart Failure, vol. 6, no. 6, pp. 1226–1232, Sep. 2019, doi: https://doi.org/10.1002/ehf2.12506.

## Gantt Chart

[Gantt Chart Link](https://docs.google.com/spreadsheets/d/1hMPUnIPTwdgqIaGhtbohadbrvBnN5f_r/edit?usp=sharing&ouid=114437293637701873553&rtpof=true&sd=true)

![Gantt Chart](GitHub_Pages/Images/CS4641_Gantt_Chart.png)

## Contribution Table

| Member   | Contributions                                                                               |
|----------|---------------------------------------------------------------------------------------------|
| Elmo     | Research and propose models, data processing methods, validation metrics.                   |
| Stan     | Dataset research and validation on Kaggle, Gantt Chart                                      |
| Sunny    | Dataset research and validation, problem and motivation set up                              |
| Shayahn  | Dataset research and validation, introduction of topic for project proposal                 |
| Joshua   | Dataset research and validation, discovering potential implementations of ML algorithms.    |
