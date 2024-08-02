# Early-Stage-Diabetes-Risk-Prediction-EDA-and-ML-Modelling

This project involves predicting the risk of early-stage diabetes in patients based on various symptoms and demographic factors. The dataset used for this project contains information on the signs and symptoms of newly diagnosed diabetic patients or those at risk of developing diabetes.

## Dataset Description

The dataset was collected through direct questionnaires administered to patients at the Sylhet Diabetes Hospital in Sylhet, Bangladesh, and approved by a doctor. The following features are included:

- `age` (Integer): Age of the patient
- `gender` (Categorical): Gender of the patient (1 for Male, 2 for Female)
- `polyuria` (Binary): Presence of polyuria (1 for Yes, 2 for No)
- `polydipsia` (Binary): Presence of polydipsia (1 for Yes, 2 for No)
- `sudden_weight_loss` (Binary): Experience of sudden weight loss (1 for Yes, 2 for No)
- `weakness` (Binary): Experience of weakness (1 for Yes, 2 for No)
- `polyphagia` (Binary): Presence of polyphagia (1 for Yes, 2 for No)
- `genital_thrush` (Binary): Presence of genital thrush (1 for Yes, 2 for No)
- `visual_blurring` (Binary): Experience of visual blurring (1 for Yes, 2 for No)
- `itching` (Binary): Experience of itching (1 for Yes, 2 for No)
- `irritability` (Binary): Presence of irritability (1 for Yes, 2 for No)
- `delayed_healing` (Binary): Experience of delayed healing (1 for Yes, 2 for No)
- `partial_paresis` (Binary): Presence of partial paresis (1 for Yes, 2 for No)
- `muscle_stiffness` (Binary): Experience of muscle stiffness (1 for Yes, 2 for No)
- `alopecia` (Binary): Presence of alopecia (1 for Yes, 2 for No)
- `obesity` (Binary): Presence of obesity (1 for Yes, 2 for No)
- `class` (Binary): Diabetes status (1 for Positive, 2 for Negative)

DATASET LINK - https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset

## Project Structure

- **Data Exploration and Visualization**: Analyzing the dataset to understand the distribution and relationships between features.
- **Preprocessing**: Encoding categorical features, scaling numerical features, and handling class imbalance using SMOTE.
- **Model Training and Evaluation**: Training various machine learning models and evaluating their performance using accuracy, classification reports, and confusion matrices.
- **Neural Network Model**: Building and training a neural network using Keras and TensorFlow.

## Models Used

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Extra Trees Classifier
- XGBoost Classifier
- Artificial Neural Network (ANN)

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- Specificity
- F1 Score

## Model Performances

The following table summarizes the performance of each model:

| Model                  | Accuracy | Precision | Recall | Specificity | F1 Score |
|------------------------|----------|-----------|--------|-------------|----------|
| Random Forest          | 0.99     | 1.00      | 0.97   | 1.00        | 0.99     |
| Gradient Boosting      | 0.98     | 1.00      | 0.94   | 1.00        | 0.97     |
| XGBoost                | 0.97     | 1.00      | 0.92   | 1.00        | 0.96     |
| Ada Boost              | 0.93     | 0.88      | 0.91   | 0.94        | 0.84     |
| Support Vector Machine | 0.95     | 0.97      | 0.89   | 0.98        | 0.93     |
| Logistic Regression    | 0.92     | 0.88      | 0.88   | 0.94        | 0.88     |
| Artificial Neural Network (ANN) | 0.92 | 0.88 | 0.88 | 0.94 | 0.88 |

## Visualization

- Distribution of categorical features
- Age distribution and boxplot
- Model performance scores and confusion matrices
- Training and validation loss and accuracy for the neural network

## Results

The trained models provide predictions on the risk of early-stage diabetes. The project compares the performance of various models and highlights the most effective ones. The confusion matrices and classification reports offer insights into each model's performance.

## Conclusion

This project demonstrates the application of various machine learning techniques and neural networks in predicting early-stage diabetes risk. The detailed analysis and visualization help in understanding the model performance and the impact of different features on the prediction.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## About Me
My Name is Sehaj Malhotra I'm a graduate student at Northeastern University, pursuing my master's in Data Analytics Engineering. I have a strong passion for Data Analytics, Data Visualization, and Machine Learning. I am an aspiring Data Analyst/Data Scientist. I love working with data to extract valuable insights.

MY LINKEDIN: https://www.linkedin.com/in/sehajmalhotra/

MY KAGGLE: https://www.kaggle.com/sehaj2001
