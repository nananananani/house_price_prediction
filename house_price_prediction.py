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

#checking for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

print("First 5 rows of dataset:")
print(data.head()) #prints first 5 rows of dataset

#separate features and target
X = data.drop("MedHouseVal", axis=1) #X will contain all columns except target house price column - bcz we want to predict house price
y = data["MedHouseVal"]  #target output we want to predict

#scaling features
scaler = StandardScaler() #mean=0, std=1
X_scaled = scaler.fit_transform(X)

#80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

models = { #making dictionary of models to try
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = [] 

for modelname, model in models.items():    #iterating through models
    model.fit(X_train, y_train)               # train model
    predictions = model.predict(X_test)       # test model

    #multiple metrics to evaluate model 
    mse = mean_squared_error(y_test, predictions) #average of sqaured differences between actual and predicted values (lower is better)
    rmse = np.sqrt(mse) #square root of mse (same unit as target variable)
    mae = mean_absolute_error(y_test, predictions) #how off predictions are on average (lower is better)
    r2 = r2_score(y_test, predictions) #variance in target explained by model (close to 1 is better)

    #storing results as list of lists
    results.append([model_name, mse, rmse, mae, r2])
    print(f"{modelname} → MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
results_df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "MAE", "R2"])

#visualizing R2 scores of models
plt.figure(figsize=(8,5))
sb.barplot(data=results_df, x="Model", y="R2", palette="viridis")
plt.title("R² Score Comparison of Models", fontsize=14)
plt.ylabel("R² Score")
plt.show()

#visualizing error metrics of models
plt.figure(figsize=(10,6))
results_melted = results_df.melt(id_vars="Model", value_vars=["MSE", "RMSE", "MAE"],var_name="Error Metric", value_name="Score")
sb.barplot(data=results_melted, x="Model", y="Score", hue="Error Metric", palette="magma")
plt.title("Error Metrics Comparison", fontsize=14)
plt.ylabel("Error Value")
plt.show()


