#importing libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

#preprocessing and model libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#loading california housing dataset
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

#checking for missing data
print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nFirst 5 rows of dataset:")
print(df.head()) #printing first 5 rows of dataset

#preprocessing
X = df.drop("MedHouseVal", axis=1)  #selecting all columns except target house price
y = df["MedHouseVal"]               #to find house price

#scaling features
scaler = StandardScaler() #mean=0, variance=1
X_scaled = scaler.fit_transform(X)

#splitting data into training and testing sets 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#defining models in a dictionary to try
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}


results = []

#iterating through each model
for model_name, model in models.items():
    model.fit(X_train, y_train) #training model
    predictions = model.predict(X_test) #testing model

    #metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    #storing results in list of lists
    results.append([model_name, mse, rmse, mae, r2])
    print(f"\nResults for {model_name}:")
    print(f"  • Mean Squared Error (MSE)       : {mse:.4f}")
    print(f"  • Root Mean Squared Error (RMSE) : {rmse:.4f}")
    print(f"  • Mean Absolute Error (MAE)      : {mae:.4f}")
    print(f"  • R-squared Score (R²)           : {r2:.4f}")

#converting results to datframe
results_df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "MAE", "R2"])


#r2
plt.figure(figsize=(8,5))
sb.barplot(data=results_df, x="Model", y="R2", hue="Model", palette="viridis", legend=False) #bargraph
plt.title("Model Comparison: R-squared Score", fontsize=14, weight='bold')
plt.ylabel("R-squared (R²) → Higher is better")
plt.xlabel("Regression Models")
plt.show()

#mse
plt.figure(figsize=(8,5))
sb.barplot(data=results_df, x="Model", y="MSE", hue="Model", palette="mako", legend=False)
plt.title("Model Comparison: Mean Squared Error", fontsize=14, weight='bold')
plt.ylabel("MSE → Lower is better")
plt.xlabel("Regression Models")
plt.show()

#rmse
plt.figure(figsize=(8,5))
sb.barplot(data=results_df, x="Model", y="RMSE", hue="Model", palette="crest", legend=False)
plt.title("Model Comparison: Root Mean Squared Error", fontsize=14, weight='bold')
plt.ylabel("RMSE → Lower is better")
plt.xlabel("Regression Models")
plt.show()

#mae
plt.figure(figsize=(8,5))
sb.barplot(data=results_df, x="Model", y="MAE", hue="Model", palette="rocket", legend=False)
plt.title("Model Comparison: Mean Absolute Error", fontsize=14, weight='bold')
plt.ylabel("MAE → Lower is better")
plt.xlabel("Regression Models")
plt.show()

