# Heart Attack Risk Prediction

### Predict the risk of heart disease using Machine Learning

This project uses a **K-Nearest Neighbors (KNN) classifier** to predict the risk of heart attack based on various health indicators such as age, blood pressure, cholesterol level, and lifestyle factors. Built as a part of the **CBSE Class XII Computer Science Project (2022–2023)**, this model demonstrates how data-driven decisions can aid early diagnosis and preventive healthcare.

---

## Table of Contents

* [Introduction](#introduction)
* [Problem Statement](#problem-statement)
* [Objective](#objective)
* [Dataset](#dataset)
* [Technologies Used](#technologies-used)
* [Model Workflow](#model-workflow)
* [Results](#results)
* [Applications](#applications)
* [Limitations](#limitations)
* [Future Work](#future-work)
* [References](#references)

---

## Introduction

Heart disease is one of the leading causes of death globally. Early detection and risk prediction can help individuals take preventive actions and seek timely medical intervention.
This project implements a machine learning approach to **predict whether a person is at risk of a heart attack** based on medical and lifestyle parameters.

---

## Problem Statement

Decisions regarding heart disease diagnosis are often based on doctors’ intuition and experience rather than data-driven insights. This can lead to errors, biases, and higher medical costs.

The project aims to build a **data-driven prediction model** to assess heart attack risk accurately and efficiently.

---

## Objective

To develop a model that predicts the likelihood of a heart attack using key features such as:

* Age, Sex, Blood Pressure, and Cholesterol
* Blood Sugar, ECG results, and Maximum Heart Rate
* Lifestyle indicators: Stress, Alcohol, Tobacco, and BMI

---

## Dataset

The dataset includes patient health records with the following attributes:

| Feature                               | Description                      |
| ------------------------------------- | -------------------------------- |
| `age`                                 | Age in years                     |
| `sex`                                 | 0 = Female, 1 = Male             |
| `cp`                                  | Chest pain type                  |
| `trestbps`                            | Resting blood pressure           |
| `chol`                                | Serum cholesterol (mg/dl)        |
| `fbs`                                 | Fasting blood sugar (>120 mg/dl) |
| `restecg`                             | ECG results                      |
| `thalach`                             | Maximum heart rate               |
| `exang`                               | Exercise-induced angina          |
| `oldpeak`                             | ST depression                    |
| `slope`, `ca`, `thal`                 | Cardiac test indicators          |
| `tobacco`, `alcohol`, `stress`, `bmi` | Lifestyle features               |
| `output`                              | 0 = No risk, 1 = Risk present    |

---

## Technologies Used

* **Python 3.10+**
* **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
* **Scikit-learn** (KNN, Random Forest, SVM)
* **Jupyter Notebook (Anaconda)**

---

## Model Workflow

1. **Data Preprocessing**

   * Load and clean dataset
   * Handle missing/structured data
   * Feature scaling and encoding

2. **Exploratory Data Analysis (EDA)**

   * Visualize feature correlations and distributions
   * Identify important predictors

3. **Model Training & Evaluation**

   * Split data into training and test sets
   * Train KNN classifier
   * Tune hyperparameters for optimal accuracy
   * Evaluate using accuracy, confusion matrix, and classification report

4. **Prediction Interface**

   * User can input patient parameters
   * Model outputs *“At Risk”* or *“Not at Risk”*

---

## Results

* Achieved high prediction accuracy with KNN (`k=19`)
* Model effectively classified high-risk and low-risk individuals
* Visual analysis confirmed meaningful correlations among health factors

---

## Applications

* Early risk detection for patients
* Health tracking & monitoring tool
* Hospital/Clinic integration for preliminary screening

---

## Limitations

* Manual data entry is required
* Accuracy depends on input data quality
* Requires several health parameters (~15+)

---

## Future Work

* Build a web-based interactive interface
* Implement real-time prediction with minimal user input
* Integrate IoT/wearable sensor data for live health monitoring
* Use deep learning models for improved accuracy

---

## References

1. Arora, S. – *Computer Science with Python (Class XI & XII)*
2. [Towards Data Science: Heart Disease Risk Assessment](https://towardsdatascience.com/heart-disease-risk-assessment-using-machine-learning-83335d077dad)
3. [Hindawi Journal: Heart Disease Prediction Using ML](https://www.hindawi.com/journals/cmmm/2022/6517716/)
4. [Kaggle: Heart Disease Dataset](https://www.kaggle.com/learn)
5. [ScienceDirect Research Articles](https://www.sciencedirect.com/science/article/pii/S1877050920315210)

---

## License

This project is licensed under the MIT License — you are free to use, modify, and distribute it, provided that proper credit is given to the original author.
