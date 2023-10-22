#!pip install pgmpy

import numpy as np
import csv
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Read the Cleveland Heart Disease data from a CSV file
heartDisease = pd.read_csv('heart.csv')

# Replace missing values with NaN
heartDisease = heartDisease.replace('?', np.nan)

# Display a few examples from the dataset
print('Few examples from the dataset are given below')
print(heartDisease.head())

# Define the structure of the Bayesian network using the column names
model = BayesianModel([
    ('Age', 'RestBP'),
    ('Age', 'Fbs'),
    ('Sex', 'RestBP'),
    ('ExAng', 'RestBP'),
    ('RestBP', 'AHD'),
    ('Fbs', 'AHD'),
    ('AHD', 'RestECG'),
    ('AHD', 'MaxHR'),
    ('AHD', 'Chol')
])

# Learning CPDs using Maximum Likelihood Estimators
print('\n Learning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Inferencing with Bayesian Network
print('\n Inferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

# Computing the Probability of Heart Disease given Age
print('\n 1. Probability of Heart Disease given Age=50')
q = HeartDisease_infer.query(variables=['AHD'], evidence={'Age': 50})
result = q.values
print(result)

# Computing the Probability of Heart Disease given cholesterol
print('\n 2. Probability of Heart Disease given cholesterol=250')
q = HeartDisease_infer.query(variables=['AHD'], evidence={'Chol': 250})
result = q.values
print(result)
