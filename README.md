# Diabetes Prediction using Machine Learning

This project predicts whether a person is diabetic based on medical diagnostic measurements from the PIMA Indians Diabetes Dataset.

-------------------------------------------------------

## Project Features

This project includes:

- Data loading and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature scaling using StandardScaler  
- Training multiple ML models (SVM, Logistic Regression, Random Forest)  
- Model evaluation and accuracy comparison  
- Choosing the best model (Logistic Regression)  
- Saving the final model and scaler using pickle  

-------------------------------------------------------

## Dataset

Source: PIMA Indians Diabetes Dataset  
File: `diabetes.csv`  

-------------------------------------------------------

## Machine Learning Models Used

- Support Vector Machine (SVM)
- Logistic Regression (final chosen model)  
- Random Forest Classifier  

Logistic Regression performed the best on the test dataset with stable accuracy, so it was selected as the final model.

-------------------------------------------------------

## Repository Structure

Diabetes-Prediction-Using-ML/
│── Diabetes Prediction.ipynb
│── diabetes.csv
│── diabetes_model.sav
│── scaler.sav
│── requirements.txt
│── README.md

-------------------------------------------------------

## How to Use the Saved Model

```python
import pickle
import numpy as np

model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

input_data = (4,110,92,0,0,37.6,0.191,30)
input_array = np.asarray(input_data).reshape(1, -1)
scaled_input = scaler.transform(input_array)

prediction = model.predict(scaled_input)

if prediction[0] == 1:
    print("Diabetic")
else:
    print("Not Diabetic")


## Tech Stack

Python
NumPy
Pandas
Scikit-learn

## Author
Adnan Shami
