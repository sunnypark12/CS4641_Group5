## CS4641_Group5
Patcharapong Aphiwetsa, SeungTaek(Stan) Lee, Shayahn Mirfendereski, Sunho(Sunny) Park, Joshua K Thomas

## Introduction/Background
According to the CDC [1], heart failure is the leading cause of death in the world, affecting over 600,000 people in the U.S. alone. There are more than a handful of machine learning models [2], [3], such as K Nearest Neighbor, Naive Bayes Classifier, and Ridge Classifier, that have been applied to predict heart failure by utilizing inputted clinical features. These features can include age, cholesterol levels, and blood pressure. The provided datasets contain comprehensive statistics of clinical factors relating to the risk of heart failure:
1. **[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) by fedesoriano:**
   - Features: Common event features caused by CVDs
2. **[Heart Failure Clinical Records](https://www.kaggle.com/datasets/nimapourmoradi/heart-failure-clinical-records) by Nima Pourmoradi:**
   - Features: Clinical records specific to heart failure
3. **[Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) by KAMIL PYTLAK:**
   - Features: Data collected from CDC, including all the major risk factors for heart failure
<br>
<br>

## Problem/Motivation
Despite significant advances in medical technology, accurately predicting heart failure remains a challenging task. Many existing models are not generalizable across different patient populations and often fail to provide actionable insights for healthcare providers. The complexity and variability of human physiology necessitate more sophisticated predictive models that can handle diverse and high-dimensional data. For instance, several research papers like [4] have identified various factors, including environmental, genetic, and lifestyle factors, that are correlated with heart failure risk, underscoring the need for improved predictive models that effectively capture and embody the relationships among them. Such models are crucial for identifying high-risk patients earlier and more accurately in a wider range of scope, enhancing treatment plans and reducing mortality rates. Leveraging machine learning models can provide robust, reliable predictions in clinical settings.
<br>

## Data Cleaning
Our group decided to utilize the first dataset we found from Kaggle, the _Heart Failure Prediction Dataset_. The dataset contained some missing values and inconsistencies that needed to be addressed before we could proceed with our analysis. Here, we describe the steps we took to clean the dataset and provide a comparison of the dataset before and after cleaning.
<br>

**Cleaning Proces: Handling Missing Values**:
   - We identified that the dataset had several missing values, particularly in the `RestingBP` and `Cholesterol` columns. These missing values were addressed by removing rows with zero values in these columns to ensure the accuracy and integrity of our data.
<br>

**Cleaning Code**

- We attempted using K-Means, but had lower accuracy, so we are planning on improving the code for the next step.
![CleaningCode](GitHub_Pages/Images/CleaningCode.png)

**Before Cleaning**

![Before_Cleaning](GitHub_Pages/Images/Before_Cleaning.png)

**After Cleaning**

![After_Cleaning](GitHub_Pages/Images/After_Cleaning.png)
As illustrated, the data cleaning process significantly improved the dataset quality, making it more suitable for subsequent analysis and modeling.
<br>

## Methods
1. **Supervised Methods:**
   - **Random Forest Classification (sklearn)** for highlighting important features and complex relationships.
   - **KNN (scikit-learn)** for simple predictions based on similarity and visualization of the data.
   - **Neural Networks (pytorch)** for handling high-dimensional data with deep learning techniques for flexibility.
2. **Data Preprocessing:**
   - **Dimensionality Reduction with PCA** to simplify data and remove redundant features.
   - **Fill in missing data (pandas)**
<br>
<br>

## 1st Supervised Method: Random Forest Model ##

1. **Loading the Dataset:** We loaded the cleaned dataset and checked the data types to ensure proper conversion of string data to appropriate types.

2. **Handling Categorical Variables:** We used label encoding for categorical variables, as it is suitable for tree-based algorithms like Random Forest.
![LabelEncoding](GitHub_Pages/Images/LabelEncoding.png)

3. **Scaling Features:** Numerical features were scaled to standardize the data, which helps in improving model performance and stability.
![ScaleData](GitHub_Pages/Images/ScaleData.png)
<br>

**Model Training and Evaluation**

We used Stratified K-Fold cross-validation to estimate the performance of our model. 

**Stratified k-fold cross-validation :**
- For skewed datasets, such as those with 90% positive and 10% negative samples, using simple k-fold cross-validation can result in folds with only negative samples.
- In such cases, stratified k-fold cross-validation is preferred. This method ensures that the ratio of labels (e.g., 90% positive and 10% negative) remains consistent in each fold.
- By maintaining this ratio, stratified k-fold cross-validation provides more reliable and consistent evaluation metrics across all folds, regardless of the metric chosen.
<br>

   **OUTPUT**
  
   Cross-Validation ROC-AUC Scores: [0.93936966 0.93418202 0.92759119 0.93625858 0.90790899]
  
   Mean ROC-AUC Score: 0.929062086192368

  Considering that ROC-AUC Score is approximated to 92%, we could say that our model has admissible performance.
<br>

**Random Forest Model**
1. Build new data set from original data : randomly select the data while keeping the same number of rows with the original data set.
2. While we don't use all the features for training the trees, we randomly select subset of features and use only those selected for training.
3. The prediction is done by passing in a new data for all the trees generated, and choosing the majority voting.
<br>

## 2nd Supervised Method: KNN Model ##

1. **Loading the Dataset:** We loaded the cleaned dataset and checked the data types to ensure proper conversion of string data to appropriate types.

2. **Handling Categorical Variables:** We used One-hot encoding for categorical variables to avoid the distorting distance metric in KNN.
![OneHotEncoding](GitHub_Pages/Images/onehotencoding.png)

3. **Reducing the Dimensions:** We ran PCA on the dataset to mitigate the impact of high dimensionality.
![PCA](GitHub_Pages/Images/PCA.png)

The heatmap below visualizes how much each original feature contributes to the principal components, with higher absolute loadings representing more influence in principal components.
![PCAComponentLoadings](GitHub_Pages/Images/PCAcomponentloadings.png)
<br>


## Results/Discussion ##

**Random Forest Classification**

By creating a heatmap, we were able to determine the optimal hyperparameters for our Random Forest model: **100 Random Decision Trees(N_estimator)**

![Heatmap](GitHub_Pages/Images/hyperparameter_heatmap.png)
<br>

**Evaluation**
The model achieved the following performance metrics:

   | Class | Precision | Recall | F1-score | Support |
   |-------|-----------|--------|----------|---------|
   | 0     | 0.84      | 0.96   | 0.89     | 71      |
   | 1     | 0.96      | 0.84   | 0.89     | 79      |
   | **Accuracy**     |       |        | 0.89     | 150     |
   | **Macro avg**    | 0.90  | 0.90   | 0.89     | 150     |
   | **Weighted avg** | 0.90  | 0.89   | 0.89     | 150     |

   **Model Accuracy:** 89.33%
<br>

The confusion matrix representing our precision, Recall, F1-score is as shown below:

![Confusion Matrix](GitHub_Pages/Images/confusionmatrix.png)
<br>
**Precision:** The model achieved a precision of 0.84. This means that 84% of the instances predicted as positive (heart disease) by the model are indeed positive. This high precision indicates that the model has a low false positive rate, which is crucial for minimizing the misclassification of healthy individuals as having heart disease.

**Recall:** The model achieved a recall of 0.96. This means that 96% of the actual positive cases (individuals with heart disease) were correctly identified by the model. This high recall value demonstrates the model's effectiveness in identifying nearly all patients who have heart disease, which is essential for ensuring that high-risk individuals are detected and can receive appropriate medical attention.

**F1-Score:** The F1-score, which is the harmonic mean of precision and recall, is 0.89. This score represents a balanced measure of the model's accuracy, taking both precision and recall into account. An F1-score of 0.89 indicates that the model performs well in both detecting heart disease and minimizing false alarms, making it a reliable tool for this predictive task.
<br>

**Comparing Random Forest Accuracy Using Feature Elimination**

Plot of Feature Importance
![feature](GitHub_Pages/Images/feature.jpeg)
<br>

Recursive Feature Elimination Graph
![graph](GitHub_Pages/Images/graph.jpeg)

For Random Forest Models, using all features results to maximum performance. 

According to the graphs, we can conclude that in our Random Forest models, eliminating features has approximate linear decrease in terms of model accuracy. 
<br>
<br>
<br>

**KNN**


<br>

## References
[1] “FastStats,” Leading Causes of Death. https://www.cdc.gov/nchs/fastats/leading-causes-of-death.htm

[2] J. Wang, “Heart Failure Prediction with Machine Learning: A Comparative Study,” Journal of Physics. Conference Series, vol. 2031, no. 1, p. 012068, Sep. 2021, doi: 10.1088/1742-6596/2031/1/012068.

[3] M. Badawy, N. Ramadan, and H. A. Hefny, “Healthcare predictive analytics using machine learning and deep learning techniques: a survey,” Journal of Electrical Systems and Information Technology, vol. 10, no. 1, Aug. 2023, doi: 10.1186/s43067-023-00108-y.

[4] V. Escolar et al., “Impact of environmental factors on heart failure decompensations,” ESC Heart Failure, vol. 6, no. 6, pp. 1226–1232, Sep. 2019, doi: https://doi.org/10.1002/ehf2.12506.
<br>

## Gantt Chart
[Gantt Chart Link](https://docs.google.com/spreadsheets/d/1hMPUnIPTwdgqIaGhtbohadbrvBnN5f_r/edit?usp=sharing&ouid=114437293637701873553&rtpof=true&sd=true)
![Gantt Chart](GitHub_Pages/Images/CS4641_Gantt_Chart.png)


## Contribution Table

<table>
  <tr>
    <th style="text-align:center">Member</th>
    <th style="text-align:center">Contributions</th>
  </tr>
  <tr>
    <td style="text-align:center">Elmo</td>
    <td style="text-align:center">KMeans Data Cleaning/Encoding, Random Forest Analysis, PCA Visualizations</td>
  </tr>
  <tr>
    <td style="text-align:center">Stan</td>
    <td style="text-align:center">Data Analysis, Random Forest Code Implementation, Visualization of Quantitative Metrics</td>
  </tr>
  <tr>
    <td style="text-align:center">Sunny</td>
    <td style="text-align:center">Data Cleaning, Random Forest Code Implementation, Github Page Design & Update</td>
  </tr>
  <tr>
    <td style="text-align:center">Shayahn</td>
    <td style="text-align:center">Random Forest Analysis, PCA Code Implementation, Github Page Documentation</td>
  </tr>
  <tr>
    <td style="text-align:center">Joshua</td>
    <td style="text-align:center">Random Forest Analysis/Visualization, PCA Code Implementation</td>
  </tr>
</table>

