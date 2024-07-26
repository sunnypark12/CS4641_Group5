# CS4641_Group5
Patcharapong Aphiwetsa, SeungTaek(Stan) Lee, Shayahn Mirfendereski, Sunho(Sunny) Park, Joshua K Thomas

# Introduction/Background
According to the CDC [1], heart failure is the leading cause of death in the world, affecting over 600,000 people in the U.S. alone. There are more than a handful of machine learning models [2], [3], such as K Nearest Neighbor, Naive Bayes Classifier, and Ridge Classifier, that have been applied to predict heart failure by utilizing inputted clinical features. These features can include age, cholesterol levels, and blood pressure. The provided datasets contain comprehensive statistics of clinical factors relating to the risk of heart failure:
1. **[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) by fedesoriano:**
   - Features: Common event features caused by CVDs
2. **[Heart Failure Clinical Records](https://www.kaggle.com/datasets/nimapourmoradi/heart-failure-clinical-records) by Nima Pourmoradi:**
   - Features: Clinical records specific to heart failure
3. **[Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) by KAMIL PYTLAK:**
   - Features: Data collected from CDC, including all the major risk factors for heart failure
<br>
<br>

# Problem/Motivation
Despite significant advances in medical technology, accurately predicting heart failure remains a challenging task. Many existing models are not generalizable across different patient populations and often fail to provide actionable insights for healthcare providers. The complexity and variability of human physiology necessitate more sophisticated predictive models that can handle diverse and high-dimensional data. For instance, several research papers like [4] have identified various factors, including environmental, genetic, and lifestyle factors, that are correlated with heart failure risk, underscoring the need for improved predictive models that effectively capture and embody the relationships among them. Such models are crucial for identifying high-risk patients earlier and more accurately in a wider range of scope, enhancing treatment plans and reducing mortality rates. Leveraging machine learning models can provide robust, reliable predictions in clinical settings.
<br>

# Data Cleaning
Our group decided to utilize the first dataset we found from Kaggle, the _Heart Failure Prediction Dataset_. The dataset contained some missing values and inconsistencies that needed to be addressed before we could proceed with our analysis. Here, we describe the steps we took to clean the dataset and provide a comparison of the dataset before and after cleaning.
<br>

**Cleaning Process: Handling Missing Values**
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

# Methods
1. ## Supervised Methods: ##
   - **Random Forest Classification (sklearn)** for highlighting important features and complex relationships.
   - **K-Nearest Neighbors (scikit-learn)** for simple predictions based on similarity and visualization of the data.
   - **Neural Network (pytorch)** for handling high-dimensional data with deep learning techniques for flexibility.
2. ## Data Preprocessing: ##
   - **Dimensionality Reduction with PCA** to simplify data and remove redundant features.
   - **Fill in missing data (pandas)**
<br>
<br>

# Supervised Method 1: Random Forest Model #

1. **Loading the Dataset:** <br> We loaded the cleaned dataset and checked the data types to ensure proper conversion of string data to appropriate types.
2. **Handling Categorical Variables:** <br> We used label encoding for categorical variables, as it is suitable for tree-based algorithms like Random Forest.
![LabelEncoding](GitHub_Pages/Images/LabelEncoding.png)
3. **Scaling Features:** <br> Numerical features were scaled to standardize the data, which helps in improving model performance and stability.
![ScaleData](GitHub_Pages/Images/ScaleData.png)
<br>

### Model Training and Evaluation ###

For Random Forest Model, we go through the following steps for training and prediction:
1. Build new data set from original data : randomly select the data while keeping the same number of rows with the original data set.
2. While we don't use all the features for training the trees, we randomly select subset of features and use only those selected for training.
3. The prediction is done by passing in a new data for all the trees generated, and choosing the majority voting.

We used Stratified K-Fold cross-validation to estimate the performance of our model. 

### Stratified k-fold cross-validation ###
- For skewed datasets, such as those with 90% positive and 10% negative samples, using simple k-fold cross-validation can result in folds with only negative samples.
- In such cases, stratified k-fold cross-validation is preferred. This method ensures that the ratio of labels (e.g., 90% positive and 10% negative) remains consistent in each fold.
- By maintaining this ratio, stratified k-fold cross-validation provides more reliable and consistent evaluation metrics across all folds, regardless of the metric chosen.
<br>

   #### OUTPUT ####
  
   Cross-Validation ROC-AUC Scores: [0.93936966 0.93418202 0.92759119 0.93625858 0.90790899]
  
   Mean ROC-AUC Score: 0.929062086192368

  Considering that ROC-AUC Score is approximated to 92%, we could say that our model has admissible performance.
<br>
<br>
<br>

# Supervised Method 2: K-Nearest Neighbors Model #

1. **Loading the Dataset:** <br> We loaded the cleaned dataset and checked the data types to ensure proper conversion of string data to appropriate types.
2. **Handling Categorical Variables:** <br> We used One-hot encoding for categorical variables to avoid the distorting distance metric in KNN.
![OneHotEncoding](GitHub_Pages/Images/onehotencoding.png)
3. **Reducing the Dimensions:** <br> We ran PCA on the dataset to mitigate the impact of high dimensionality.
![PCA](GitHub_Pages/Images/PCA.png)

The heatmap below visualizes how much each original feature contributes to the principal components, with higher absolute loadings representing more influence in principal components.

![PCAComponentLoadings](GitHub_Pages/Images/PCAcomponentloadings.png)
<br>

### Model Training and Evaluation ### 

K-Nearest Neighbors is a type of instance-based learning (also known as lazy learning), which has a different approach to training and prediction compared to other machine learning models.

#### Training in KNN: ####

* No Explicit Training Phase: KNN does not have a traditional training phase where the model parameters are learned from the data.
* Storage of Training Data: The "training" in KNN essentially involves storing the entire training dataset. There are no weights or parameters to update.
* Distance Computation: During prediction, KNN computes the distance between the query point and all points in the training dataset to find the k-nearest neighbors.

#### Prediction in KNN: ####

* Distance Calculation: For a given query point, KNN calculates the distance (commonly Euclidean distance) to all training points.
* Neighbor Selection: It selects the k-nearest neighbors based on the smallest distances.
* Majority Voting: In classification, KNN assigns the class that is most common among the k-nearest neighbors. In regression, it takes the average of the neighbors' values.
<br>
<br>

# Supervised Method 3: Neural Network #

1. **Loading the Dataset:** <br> We loaded the cleaned dataset and checked the data types to ensure proper conversion of string data to appropriate types.

2. **Data Preprocessing:** <br> We preprocessed the data to prepare it for training the neural network, ensuring that our data is in correctly formatted and scaled for the neural network to learn effectively. This involves several steps:

   1. **One-Hot Encoding Categorical Features**
      We used `OneHotEncoder` to convert categorical features into numerical values.
   2. **Concatenating Numerical and Encoded Categorical Data**
      We combined the numerical features with the one-hot encoded categorical features.
   3. **Splitting Data into Training and Testing Sets**
      We used `train_test_split` to split the data into training and testing sets.
   4. **Standardizing the Data**
      We scaled the features using `StandardScaler` to standardize the data.
   5. **Converting Data to PyTorch Tensors**
      Finally, we converted the NumPy arrays into PyTorch tensors, which are required for training the neural network.
   
   **Understanding the Structure of Training Data: Visualization Using PCA** <br>
   Though we didn't use PCA during training, the visualization of training data with PCA provided better understanding of the data structure and insights on separability of classes. We found out reducing our training data to a 2D projection would be effective in having better interpretation.
   
   ![PCA_NN](GitHub_Pages/Images/PCA_NN.png)
   
   - Class Separation: Two classes(0 and 1) shown in different colors have noticeable but not complete separation, indicating a positive sign of classification in that the classes are somewhat distinguishable.
   - Cluster Density: The density of points for each class varies across the plot, but there are regions where the two classes overlap significantly, which could lead to some misclassifications.
   - Principal Components: The first principal component (PC1) seems to capture more variance as there is more spread along the x-axis compared to the y-axis (PC2).
   - Implications for Model Training: The overlap between classes indicates that while some regions of the feature space are well-separated, others are not; thus suggesting that a more complex model might be needed to capture the nuances in the data.

4. **Defining the Neural Network:** <br> Using the `SimpleNN` class in the PyTorch framework, we defined a simple feedforward neural network using the following steps:

   1. **Initialization**: (__init__ method)
   2. **Input Parameters**:
       * input_size: The number of input features
       * num_layers: The number of hidden layers
       * hidden_size: The number of neurons in each hidden layer
       * function: The activation function to use ('ReLU' or 'Sigmoid')
   3. **Network Layers**:
       * self.proj: A linear layer that projects the input features to the hidden size
       * self.layers: A sequential container to hold multiple hidden layers. Each hidden layer consists of a linear transformation followed by an activation function (ReLU or Sigmoid)
       * self.output: A linear layer that projects the hidden layer output to the number of classes (2 in this case, for binary classification)
   4. **The Forward Method**: The forward method defines how the input data passes through the network
       * Projection: The input x is first passed through the projection layer (self.proj)
       * Hidden Layers: The projected input is then passed through the sequential container of hidden layers (self.layers)
       * Output Layer: Finally, the output from the hidden layers is passed through the output layer (self.output) to produce the final logits for classification
<br>

### Model Training and Evaluation ###

The train_model function is designed to train a neural network model with specified hyperparameters and return the training and validation losses and accuracies over epochs. **The function also includes early stopping to prevent overfitting**.

#### 1. Initialization ####

1. **Model Initialization:**
   * The model is initialized with the given input_size, num_layers, hidden_size, and activation function, and then moved to the specified device (CPU or GPU).
2. **Optimizer and Loss Function:**
   * The Adam optimizer is used with a learning rate specified in the hyperparameters.
   * nn.CrossEntropyLoss() is the loss function used, which is suitable for classification tasks.
   * The Adam optimizer is used to optimize the internal parameters (weights and biases) of the neural network during the learning process.
3. **Data Loaders:**
   * The training and testing datasets are loaded into DataLoaders with the specified batch size for efficient batch processing during training and evaluation.
4. **Tracking Variables:**
   * Lists to store training and validation losses and accuracies for each epoch.
   * Variables for early stopping: best_test_loss to track the best validation loss and patience_counter to count the number of epochs without improvement.

#### 2. Training Loop ####

1. **Epoch Loop:**
    * The model is set to training mode using model.train().
    * The running loss and correct predictions are tracked for each batch in the training set.
    * The optimizer is zeroed, the model performs a forward pass, the loss is calculated and backpropagated, and the optimizer steps to update the model weights.
    * Training loss and accuracy are calculated and appended to their respective lists.
2. **Validation Loop:**
    * The model is set to evaluation mode using model.eval().
    * The loss and correct predictions are tracked for each batch in the validation set.
    * Validation loss and accuracy are calculated and appended to their respective lists.
3. **Early Stopping:**
    * If the current validation loss is better than the previous best validation loss minus a specified delta, the best validation loss is updated, and the patience counter is reset.
    * If no improvement is seen for a number of epochs equal to params['patience'], training stops early.
  
#### 3. Return Values ####

The function returns the training and validation losses and accuracies over epochs, and the trained model.
<br>

#### 4. Hyperparameter Tuning ####

In the context of neural networks, hyperparameter tuning can be resource-intensive due to the computational demands of training these models. Cross-validation, although highly effective for many machine learning models, is less practical for neural networks because training these models is typically very time-consuming.

Thus, we used combination of grid search and early stopping for hyperparameter tuning:
- Grid Search: Trying out combinations of different hyperparameters
- Early Stopping: Monitoring validation loss and stopping training when performance ceases to improve to avoid overfitting

### Hyperparameters ###
**1. num_layers (Number of Layers):** <br>Number of hidden layers in the neural network<br>More layers can potentially capture more complex patterns in the data but can also lead to overfitting. Increasing the number of layers might improve performance up to a certain point, beyond which the model might overfit.<br>
**2. hidden_size (Number of Neurons per Layer):** <br>Number of neurons in each hidden layer<br>More neurons can increase the model's capacity to learn from data but can also lead to overfitting. Increasing the number of neurons may improve performance up to a point, after which it could lead to overfitting.<br>
**3. activation (Activation Function):** <br>Introduces non-linearity into the model, enabling it to learn complex patterns<br>ReLU might perform better for deeper networks as it mitigates the vanishing gradient problem while Sigmoid might be useful for shallower networks or specific tasks where its output range is more appropriate.<br>
**4. learning_rate (Learning Rate):** <br>Controls how much the model's weights are adjusted with respect to the loss gradient during training<br> Lower learning rate might lead to more stable convergence but may require more epochs to train while higher learning rate might speed up training but could cause the model to miss the optimal solution.<br>
**5. batch_size (Batch Size):** <br>Number of samples processed before the model's internal parameters are updated<br>Smaller batch sizes can lead to noisier updates but more frequent adjustments while larger batch sizes can make training more stable but might lead to overfitting.<br>
**6. epochs (Number of Epochs):** <br>Number of complete passes through the training dataset<br>More epochs generally improve performance up to a point, after which the model might start overfitting.<br>
**7. patience:** <br>Number of epochs with no improvement after which training will be stopped<br>
**8. delta:** <br>minimum change in the monitored quantity to qualify as an improvement.<br>

![Hyperparameter](GitHub_Pages/Images/Hyperparam.png)
<br>
<br>

# Results/Discussion #

## 1. Random Forest Classification ##

By creating a heatmap, we could determine optimal hyperparameters for Random Forest model: **100 Random Decision Trees(N_estimator)**

![Heatmap](GitHub_Pages/Images/hyperparameter_heatmap.png)
<br>

### Evaluation ###
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

#### Comparing Random Forest Accuracy Using Feature Elimination ####

##### Plot of Feature Importance ##### 
![feature](GitHub_Pages/Images/feature.jpeg)
<br>

##### Recursive Feature Elimination Graph #####
![graph](GitHub_Pages/Images/graph.jpeg)

For Random Forest Models, using all features results to maximum performance. 

According to the graphs, we can conclude that in our Random Forest models, eliminating features has approximate linear decrease in terms of model accuracy. 
<br>
<br>
<br>

## 2. K-Nearest Neighbors ##

We found the optimal hyper parameter k using cross-validation which gives us an unbiased estimate of the model's performance. Cross-validation is the average accuracy of the model on the validation folds during the cross-validation proces. This helps estimate the model's performance on unseen data.

![OptimalK](/GitHub_Pages/Images/KNNoptimalK.png)

From the graph we can conclude that k = 23 is the optimal number of neighbors.

While k = 23 gives us the optimal k value computed by using cross-validation, we found out that using different k values sometimes gives us better test accuracies. We explored this phenomenon by evaluating the model accuracy through testing with different k values ranging from 1 to 30.

![NumberOfNeighborsVsAccuracy](GitHub_Pages/Images/NumNeighborsVsAccuracy.png)

As shown in the plot, in some cases using other values of k besides k = 23 leads to having better test set accuracy.

Generally if model performs well during cross-validation, it is expected to perform well on the test set because cross-validation helps mitigate overfitting. Higher test set accuracy indicates better generalization, but it is also possible for the test accuracy to differ from the cross-validation due to the nature of data splits.

To elaborate, some possible explanations of why this is happening could be of the following reasons:
1. Randomness in Data Splits: The specific split of the training and test data can lead to variations in accuracy.
2. Overfitting: A model might overfit to the validation folds during cross-validation, resulting in slightly lower performance on the test set.
3. Sample Size: With limited data, slight variations in the data splits can have a noticeable impact on performance metrics.

Thus, the practical approach would be to choose a k value based on the cross-validation since it is a more reliable estimate of performance across different data splits. 
<br>

### Evaluation ###
The model achieved the following performance metrics:

   | Class | Precision | Recall | F1-score | Support |
   |-------|-----------|--------|----------|---------|
   | 0     | 0.88      | 0.90   | 0.89     | 71      |
   | 1     | 0.91      | 0.89   | 0.90     | 79      |
   | **Accuracy**     |       |        | 0.89     | 150     |
   | **Macro avg**    | 0.89  | 0.89   | 0.89     | 150     |
   | **Weighted avg** | 0.89  | 0.89   | 0.89     | 150     |

   **Model Accuracy:** 89%
<br>

The confusion matrix representing our precision, Recall, F1-score is as shown below:

![Confusion Matrix](GitHub_Pages/Images/KNNconfusionmatrix.png)
<br>

The model demonstrates a high level of accuracy, indicating that it performs well in classifying instances correctly.
The precision and recall values for both classes are high, indicating that the model is both accurate in its predictions and effective at identifying instances of each class.

* **Class 0 Analysis:** The precision (88%) and recall (90%) for class 0 are both high, suggesting that the model is effective at correctly identifying instances of class 0 and minimizing false positives.

* **Class 1 Analysis:** The precision (91%) and recall (89%) for class 1 are also high, indicating that the model accurately identifies instances of class 1 and has a low false-negative rate.
    
* **Confusion Matrix Insights:** The confusion matrix shows a balanced performance between the two classes, with slightly more false positives for class 0 (7) than false negatives for class 1 (9). The number of true positives is high for both classes, showing the model's effectiveness.
<br>

## 3. Neural Network ##
The final performance of our neural network model was evaluated using the best hyperparameters identified through an extensive hyperparameter tuning process. The training, validation, and testing losses and accuracies were monitored and plotted over the course of 25 epochs.

**1. Hyperparameter Tuning Results**<br>![Hyperparameter Tuning Results](GitHub_Pages/Images/hyperResult.png)<br>Best Parameters: 'num_layers': 12, 'hidden_size': 256, 'activation': 'ReLU', 'learning_rate': 0.001, 'batch_size': 16, 'epochs': 50, 'val_accuracy': 0.9107142857142857, 'delta': 0.001, 'patience': 5<br><br>
**2. Training, Validation, Testing Loss**<br>Early stopping triggered at epoch 10<br>Test Accuracy: 0.8839<br>![NNLoss](GitHub_Pages/Images/NNLoss.png)

**Plot Above** displays the training, validation, and testing losses across 10 epochs due to early stopping:<br>
1. Early and Mid Phases (Epochs 1-7):<br>The training, validation, and testing losses all decrease, indicating that the model is learning effectively. The validation and testing losses decrease at a similar rate to the training loss, suggesting good generalization during these phases.
4. Later Phases (Epochs 8-10):<br>The validation and testing losses start to increase, while the training loss continues to decrease slightly. This divergence is a classic sign of overfitting, where the model starts to memorize the training data rather than generalizing to unseen data.

**Plot Below** displays the training, validation, and testing accuracies over the same epochs:<br>
1. Early and Mid Phases (Epochs 1-7):<br>Training, validation, and testing accuracies increase significantly, with all three metrics following a similar trend. This indicates that the model is improving its performance on both seen (training) and unseen (validation and testing) data.
4. Later Phases (Epochs 8-10):<br> The training accuracy remains high and stable. The validation and testing accuracies show slight fluctuations and do not improve further. In fact, they start to diverge from the training accuracy, suggesting that the model is beginning to overfit.

#### Early Stopping ####
Early stopping criteria were employed during training to prevent overfitting. The model was set to stop training if there was no significant improvement in validation loss for a certain number of epochs (patience: 5).

#### Model Performance: ####
The final model achieves a high validation accuracy of around 90%, indicating strong generalization performance.
<br>

#### Overfitting Mitigation: #### 
The use of early stopping, appropriate activation functions, and hyperparameter tuning helped mitigate overfitting and ensured the model remained robust. We can conclude that to mitigate overfitting our model should conduct early stopping around epoch 7.
<br>


# Comparison of Models #
Here is a comparison of the three supervised learning models we used in terms of various performance metrics:

### Evaluation Metrics ###
The table below shows the performance of each model in terms of accuracy, precision, recall, and F1-score:

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Random Forest         | 89.33%   | 0.84      | 0.96   | 0.89     |
| K-Nearest Neighbors   | 89%      | 0.88      | 0.90   | 0.89     |
| Neural Network        | 88.39%   | -         | -      | -        |

### Confusion Matrix Insights ###
- **Random Forest:** Demonstrated the highest recall for detecting heart disease, making it highly effective in identifying positive cases. It also had the highest overall accuracy.
- **K-Nearest Neighbors:** Showed balanced performance across precision, recall, and F1-score, indicating reliability. However, its performance can be influenced by the choice of k and data splits.
- **Neural Network:** Achieved high validation accuracy with effective overfitting mitigation(early stopping), but required extensive computational resources for training.

## Conclusion ##
All three models performed well in predicting heart disease, with slight variations in their performance metrics. The Random Forest model achieved the highest accuracy and recall, making it the most effective at identifying individuals with heart disease. The K-Nearest Neighbors model showed balanced performance across precision, recall, and F1-score, indicating its reliability. The Neural Network model also performed strongly, with effective mitigation of overfitting and robust generalization.

**Final Remarks:** 
The Random Forest model, with its highest accuracy and recall, is recommended for this specific task of heart disease prediction. However, the K-Nearest Neighbors and Neural Network models also provide valuable insights and could be further optimized for improved performance.

**Next Steps**
- **Model Optimization:** Further optimize each model, particularly focusing on feature selection and hyperparameter tuning.
- **Ensemble Methods:** Explore ensemble methods to combine the strengths of each model for improved predictive performance.
- **Larger Datasets:** Apply the models to larger and more diverse datasets to enhance generalizability and robustness.
- **Clinical Integration:** Collaborate with healthcare providers to integrate the models into clinical settings and validate their effectiveness in real-world applications.


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
    <td style="text-align:center"></td>
  </tr>
  <tr>
    <td style="text-align:center">Stan</td>
    <td style="text-align:center"></td>
  </tr>
  <tr>
    <td style="text-align:center">Sunny</td>
    <td style="text-align:center"></td>
  </tr>
  <tr>
    <td style="text-align:center">Shayahn</td>
    <td style="text-align:center">Neural Network Implementation and Visual Analysis, Model Comparisons, GitHub Documentation</td>
  </tr>
  <tr>
    <td style="text-align:center">Joshua</td>
    <td style="text-align:center"></td>
  </tr>
</table>

