#importing libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
#using california housing dataset from sklearn
from sklearn.datasets import fetch_california_housing
#implement train and test split
from sklearn.model_selection import train_test_split
#data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#loading dataset
housing_data = fetch_california_housing(as_frame=True)   # load dataset as dataframe
df = housing_data.frame
print("First 5 rows of dataset:")
print(data.head())
#separate features and target
X = data.drop("MedHouseVal", axis=1) 
y = data["MedHouseVal"]              
#scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)