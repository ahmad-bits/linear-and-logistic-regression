# Linear and Logistic Regression from Scratch

This repository contains my implementations of Linear Regression and Logistic Regression built from scratch using NumPy.

Rather than relying on high-level machine learning libraries like scikit-learn for model fitting, the goal of this project was to understand the internal mechanics of both algorithms. Writing the code manually helped me learn how cost functions are formulated, how analytical gradients are derived and calculated, how weights and bias are updated through gradient descent, and how predictions are generated.

## Project Overview

In this project, I worked through two fundamental supervised learning algorithms:

- **Linear Regression**: Used for continuous numerical prediction. The model was trained and tested on a house price dataset to predict property prices based on various house features.
- **Logistic Regression**: Used for binary classification. The model was trained and tested on a heart disease dataset to predict whether a patient has heart disease (1) or does not (0).

Both models are structured as standalone Python classes and paired with Jupyter notebooks that walk through the entire experiment, from data preprocessing to evaluation and visualization.

## What I Implemented

### Linear Regression (`linear_regression/linear_regression.py`)
- **Prediction Function**: Implemented linear predictions using matrix multiplication (`X . w + b`).
- **Cost Function with L2 Regularization**: Implemented Mean Squared Error (MSE) along with an L2 regularization penalty (Ridge) to penalize large weights and prevent overfitting.
- **Gradients**: Computed the partial derivatives of the cost function with respect to weights (`dj_dw`) and bias (`dj_db`), including the regularization term.
- **Batch Gradient Descent**: Updated weights and bias simultaneously over a specified number of iterations and learning rate, while recording the cost history at every step.
- **Target Transformation**: Tested training on log-transformed prices to handle right-skewed target values and reduce percentage error on expensive homes.

### Logistic Regression (`logistic_regression/logistic_regression.py`)
- **Sigmoid Activation**: Implemented the sigmoid function to map linear combinations into probabilities between 0 and 1, with clipping to avoid numerical overflow during exponentiation.
- **Cost Function (Binary Cross-Entropy / Log Loss)**: Implemented cross-entropy loss to measure the distance between predicted probabilities and actual labels, with epsilon clipping to avoid `log(0)`.
- **Gradients**: Calculated gradients of the log loss with respect to weights (`dw`) and bias (`db`).
- **Gradient Descent Optimization**: Updated model weights and bias across training iterations to minimize cross-entropy loss.
- **Predictions and Thresholding**: Generated probability outputs (`predict_probability`) and applied a 0.5 decision threshold to return discrete class predictions (`predict`).

## Datasets

Both datasets are stored in the `data/` directory:

1. **Housing Price Dataset (`data/housing_data.csv`)**
   - **Task**: Regression (predict house prices in dollars).
   - **Size and Features**: 545 samples and 11 input features, including area, bedrooms, bathrooms, stories, parking, and categorical features like main road access, guest room, basement, and air conditioning.
   - **Notes**: Prices are right-skewed and range widely, making feature scaling and target transformation helpful.

2. **Heart Disease Dataset (`data/heart_data.csv`)**
   - **Task**: Binary classification (predict heart disease: 0 or 1).
   - **Size and Features**: 1,190 samples and 11 features, including age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, and ST slope.
   - **Notes**: Required cleaning duplicate rows, imputing missing zero values in cholesterol, and encoding categorical variables.

## Workflow

The workflow used across both models followed a consistent, practical pipeline:

1. **Loading the Datasets**: Loaded raw CSV files with pandas to inspect feature columns, distributions, and shapes.
2. **Data Preparation and Cleaning**:
   - For housing data: separated features from the target price and handled categorical columns.
   - For heart disease data: removed duplicate patient records, imputed missing zero cholesterol values using the median of valid records, created an engineered feature (`hr_ratio` = max heart rate / (220 - age)), and one-hot encoded nominal categorical features (`chest pain type`, `resting ecg`, `ST slope`).
3. **Train-Test Split**:
   - Shuffled the data randomly with a fixed seed (`np.random.seed(42)`) before splitting to remove any order bias from pre-sorted records.
   - Split both datasets into an 80% training set and a 20% testing set.
4. **Feature Scaling**:
   - Standardized features using z-score normalization (`(X - mean) / std`) calculated solely from the training data to avoid data leakage.
   - Scaled the target variable in linear regression during training and later transformed predictions back to original dollar units for meaningful interpretation.
5. **Training with Gradient Descent**:
   - Initialized weights to zeros and bias to 0.0.
   - Ran gradient descent for a fixed number of iterations, tuning learning rates to ensure the cost steadily decreased without divergence.
   - Tracked cost values after each iteration in a history list.
6. **Making Predictions**:
   - Ran test features through the learned weights and bias.
   - For linear regression: inverted standard scaling (and exponential transformation where applicable) to evaluate dollar predictions.
   - For logistic regression: calculated probabilities with the sigmoid function and classified instances using a 0.5 cutoff.
7. **Evaluation**:
   - Evaluated Linear Regression using Mean Absolute Percentage Error (MAPE) and R-squared (R2).
   - Evaluated Logistic Regression by calculating Accuracy, Precision, Recall, and F1-score from scratch. Also inspected learned weights to identify which features had the strongest relationship with the target.

## Technologies Used

- **Python**: Primary language used for writing classes and experiment scripts.
- **NumPy**: Used for core vector math, matrix operations, gradient calculations, and cost functions.
- **Pandas**: Used for loading data, cleaning records, handling missing values, and one-hot encoding.
- **Matplotlib**: Used for plotting cost curves, scatter plots, and classification decision boundaries.
- **Jupyter Notebook**: Used for interactive development, running experiments, and documenting intermediate steps.

## How to Run

1. Clone or download this repository.
2. Install the required Python packages:
   ```bash
   pip install numpy pandas matplotlib jupyter
   ```
3. Start Jupyter Notebook and open either experiment:
   ```bash
   jupyter notebook linear_regression/linear_experiments.ipynb
   ```
   or
   ```bash
   jupyter notebook logistic_regression/logistic_experiments.ipynb
   ```
