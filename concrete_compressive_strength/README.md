# ConcreteForce: Empowering Strength Prediction with Neural Networks

![screenshot_20210904-164329](https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/0af35424-c5fa-408b-9f40-e56f940a2c91)

## Project Background
Concrete is one of the most widely used construction materials due to its durability, versatility, and affordability. The compressive strength of concrete is a critical parameter that determines its ability to withstand loads and stresses in structural applications. Traditionally, the compressive strength of concrete is determined through time-consuming and costly experimental tests, which involve casting and curing concrete samples before subjecting them to destructive testing.

With the advancements in artificial intelligence and machine learning, there is an opportunity to develop predictive models that can accurately estimate the compressive strength of concrete using non-destructive methods. Deep neural networks (DNNs) have shown remarkable capabilities in learning complex patterns from data and making accurate predictions. By leveraging DNNs, we aim to create a predictive model that can estimate the compressive strength of concrete based on its composition and curing conditions.

## Project Objectives
The primary objective of this project is to develop a deep neural network model capable of accurately predicting the compressive strength of concrete. Specifically, the project aims to:

1. Data Collection: Gather a comprehensive dataset comprising various concrete mixtures along with their corresponding compressive strength values. The dataset will include information such as the type and proportions of cement, aggregates, water-cement ratio, curing duration, and other relevant factors.

2. Data Preprocessing: Clean and preprocess the collected data to remove outliers, handle missing values, and normalize the features. This step is crucial for ensuring the quality and reliability of the dataset for training the neural network model.

3. Model Development: Design and train a deep neural network architecture suitable for regression tasks. Experiment with different network architectures, activation functions, optimizers, and hyperparameters to find the configuration that yields the best performance in predicting concrete compressive strength.

4. Model Evaluation: Evaluate the performance of the trained neural network model using appropriate metrics such as mean absolute error, mean squared error, and coefficient of determination (R-squared). Conduct rigorous testing and validation to assess the model's accuracy, robustness, and generalization capabilities.

## Data Overview
Utilizing Kaggle (https://www.kaggle.com/datasets/hamzakhurshed/concrete-strength-dataset) for comprehensive concrete dataset for predictive modeling.

<img width="400" alt="Screenshot 2024-04-20 at 7 27 38 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/9f408f8d-3ad0-40ba-a158-d42c35b3205b">

<img width="400" alt="Screenshot 2024-04-20 at 7 28 06 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/165e933f-f3a9-4e58-b08f-5856e54ccb0d">

## Data Visualization and Data Preprocessing
(A) Check for Missing Values: To ensure data completeness and integrity, facilitating accurate analysis and decision-making processes.

![output](https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/8d2640a7-4962-49b7-8ba9-53752aef2336)

(B) Visualizing the Relationship Between Ingredients and Concrete Strength: To discern potential correlations or patterns between different input features and concrete compressive strength, aiding in understanding the relationship and identifying influential factors.

![output](https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/54b39653-5cdb-4550-a445-4b9867854163)

(C) Visualizing the Kendall Correlation Between Numerical Variables: Provides insight into the strength and direction of their ordinal relationships, aiding in understanding the degree of association while accounting for potential nonlinearities.

![output](https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/ac5c0b93-9c82-4bcb-abf9-4c919271b18e)

(D) Feature Selection using the Kendall Correlation Test: To identify and retain the most relevant and least correlated features, thus improving model performance and interpretability while reducing redundancy in the dataset.

<img width="600" alt="Screenshot 2024-04-20 at 7 37 05 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/cfd001d2-c4c0-4a46-b76a-0d41c0a2c28d">

Feature Name: cement
p-value: 0.0000
p-value is less than 0.05 signficance level. There is an evidence to conclude that cement is statistically significant to csMPa

Feature Name: slag
p-value: 0.0000
p-value is less than 0.05 signficance level. There is an evidence to conclude that slag is statistically significant to csMPa

Feature Name: flyash
p-value: 0.0169
p-value is less than 0.05 signficance level. There is an evidence to conclude that flyash is statistically significant to csMPa

Feature Name: water
p-value: 0.0000
p-value is less than 0.05 signficance level. There is an evidence to conclude that water is statistically significant to csMPa

Feature Name: superplasticizer
p-value: 0.0000
p-value is less than 0.05 signficance level. There is an evidence to conclude that superplasticizer is statistically significant to csMPa

Feature Name: coarseaggregate
p-value: 0.0000
p-value is less than 0.05 signficance level. There is an evidence to conclude that coarseaggregate is statistically significant to csMPa

Feature Name: fineaggregate
p-value: 0.0000
p-value is less than 0.05 signficance level. There is an evidence to conclude that fineaggregate is statistically significant to csMPa

Feature Name: age
p-value: 0.0000
p-value is less than 0.05 signficance level. There is an evidence to conclude that age is statistically significant to csMPa

(E) Anomaly Detection using Isolation Forest: To identify and subsequently remove outliers or anomalous data points from the dataset, enhancing the quality and reliability of subsequent analysis or modeling tasks.

<img width="600" alt="Screenshot 2024-04-20 at 7 35 40 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/04239925-e691-4aef-a9eb-8d23dd5c16bd">

<img width="600" alt="Screenshot 2024-04-20 at 7 36 05 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/480ceb2b-16dc-4ee3-80b5-e6c5840ab22d">

## Data Splitting
Splitting the data into 80% for training and 20% for testing is to effectively train a machine learning model on a larger portion of the dataset while reserving a separate portion to evaluate its performance and generalization ability.

<img width="600" alt="Screenshot 2024-04-20 at 7 39 12 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/90fbc969-3bd6-4248-b86c-c052804396a0">

## Data Normalizing - StandardScaler
Normalizing input features using the standard scaler is to scale and center the data, ensuring that each feature contributes equally to the model training process, thereby improving the convergence and performance of machine learning algorithms.

<img width="600" alt="Screenshot 2024-04-20 at 7 39 53 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/7556068e-7904-4de5-a466-cd517dac8024">

<img width="600" alt="Screenshot 2024-04-20 at 7 40 23 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/a503e857-b334-44fa-8b71-6a6894ae1b0c">

## Deep Neural Network (DNN)

The architecture of the deep neural network (DNN) defined by the provided code consists of two dense layers. The first dense layer has 128 neurons and uses the Gaussian Error Linear Unit (GELU) activation function, with L2 regularization applied to the kernel weights. This layer has an input dimension of 8, corresponding to the number of input features. A dropout layer with a dropout rate of 0.1 is inserted after the first dense layer to mitigate overfitting by randomly setting a fraction of input units to zero during training. The second dense layer consists of a single neuron with ReLU activation, which outputs the final prediction. Overall, this DNN architecture employs a feedforward structure with regularization and dropout mechanisms to facilitate learning and improve generalization performance.

<img width="600" alt="Screenshot 2024-04-20 at 7 41 33 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/9fe2cca8-c3d0-4156-b643-16226c75e3cc">

Holdout Validation:

<img width="245" alt="Screenshot 2024-04-20 at 7 42 24 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/7aebd85e-74cb-4df9-adb1-f3d3a5cca943">

K-Fold Cross-Validation:

<img width="245" alt="Screenshot 2024-04-20 at 7 43 09 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/d4c2d31c-b75f-4c25-bfc9-4eab566572ab">

The results from hold-out validation, with a train R2-Score of 0.917512 and test R2-Score of 0.904730, demonstrate strong performance in predicting the target variable, albeit slightly lower on the test set, along with relatively low mean squared error (MSE) values of 20.507282 and 24.807976 for train and test sets respectively. In contrast, K-Fold cross-validation yielded slightly lower R2-Scores for both train (0.907086) and test (0.865536) sets, indicating a slight decrease in model generalization compared to hold-out validation. Additionally, the MSE values for K-Fold cross-validation were higher than those from hold-out validation, with train and test MSE of 23.080122 and 31.740225 respectively, suggesting a slightly poorer fit to the data when using this cross-validation technique.

Comparison Plot:

![output](https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/58ab0fc2-61f3-41e6-925b-1c4e8262b920)

Most of the training and testing data align along the diagonal line, it suggests that the 1D-CNN predictions closely match the actual values. This alignment signifies that the model is effectively capturing the underlying patterns and trends in the data, leading to accurate predictions. The diagonal line represents perfect prediction, where the predicted values are equal to the actual values. Therefore, the proximity of the data points to this line indicates the model's ability to generalize well and make reliable predictions across the dataset, implying a strong performance of the 1D-CNN regressor in capturing the relationships within the input data.

## 1-Dimensional Convolutional Neural Network (1D-CNN)

The architecture of the deep neural network defined by the provided code involves a one-dimensional convolutional neural network (CNN) followed by dense layers. The input data is reshaped into a three-dimensional format to fit the convolutional layer's input requirements. The CNN consists of a convolutional layer with 16 filters, each with a kernel size of 5 and GELU activation, with padding applied to maintain the input size. Subsequently, the output is flattened to be compatible with the following dense layers. The dense layers consist of 128 neurons with GELU activation and L2 regularization, followed by a dropout layer to prevent overfitting. Finally, a single neuron with ReLU activation serves as the output layer for prediction. This architecture leverages convolutional operations for feature extraction and subsequent dense layers for further processing and prediction.

<img width="600" alt="Screenshot 2024-04-20 at 7 45 41 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/195ac3ea-29ce-4835-a7ae-52fb297d7da3">

Holdout Validation:

<img width="245" alt="Screenshot 2024-04-20 at 7 46 30 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/088126ba-7de4-4304-80fc-c0d31263a195">

K-Fold Cross-Validation:

<img width="245" alt="Screenshot 2024-04-20 at 7 47 04 PM" src="https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/5fb668e2-8773-4365-a675-fe604af96b8d">

The results from hold-out validation for the 1D-CNN model indicate strong performance, with a train R2-Score of 0.925348 and test R2-Score of 0.909768, showcasing its ability to accurately predict the target variable, with relatively low mean squared error (MSE) values of 18.559225 and 23.496241 for train and test sets respectively. Conversely, K-Fold cross-validation exhibited slightly lower R2-Scores for both train (0.890999) and test (0.884620) sets, implying a slight decrease in the model's generalization compared to hold-out validation. Interestingly, despite this decrease in R2-Scores, the MSE values for both train and test sets in K-Fold cross-validation were quite similar (27.079923 for train and 27.072859 for test), indicating consistent predictive performance across folds and suggesting robustness in model fit across different subsets of the data.

Comparison Plot:

![output](https://github.com/jeffwongqy/Materials-Informatics/assets/100281127/640ce768-bda0-47a5-abe6-e6e20e6e71e9)

Most of the training and testing data align along the diagonal line, it suggests that the 1D-CNN predictions closely match the actual values. This alignment signifies that the model is effectively capturing the underlying patterns and trends in the data, leading to accurate predictions. The diagonal line represents perfect prediction, where the predicted values are equal to the actual values. Therefore, the proximity of the data points to this line indicates the model's ability to generalize well and make reliable predictions across the dataset, implying a strong performance of the 1D-CNN regressor in capturing the relationships within the input data.

## Conclusion
Based on the provided results, both the DNN and 1D-CNN models demonstrate strong predictive performance for concrete compressive strength estimation. However, the 1D-CNN model appears to have a slight edge in terms of hold-out validation performance, exhibiting higher R2-Scores and lower MSE values compared to the DNN model. While the DNN model showcases comparable performance, the 1D-CNN model's ability to capture spatial dependencies within the data, particularly relevant in concrete strength prediction where structural features play a crucial role, suggests its potential superiority in real-life applications. Additionally, the 1D-CNN model exhibits consistent predictive performance across different subsets of the data in K-Fold cross-validation, indicating robustness and reliability. Therefore, based on these findings, the 1D-CNN model would be the preferred choice for predicting concrete compressive strength in real-life scenarios.

















