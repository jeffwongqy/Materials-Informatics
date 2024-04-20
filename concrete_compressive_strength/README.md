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









