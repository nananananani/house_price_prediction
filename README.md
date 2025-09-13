# Predicting House Prices ୨୧
❀。• *₊°。 ❀°。❀。• *₊°。 ❀°。❀。• *₊°。 ❀°。❀。• *₊°。 ❀°。❀。• *₊°。 ❀°。❀。• *₊°。 ❀°。❀。• *₊°。 ❀°。❀。• *₊°
###### (recruitment task for ACM SIGKDD)
### 1. Overview
This project predicts house prices in California using the California Housing Dataset from sklearn.
It compares the performance of three regression models:
   * Linear Regression
   * Decision Tree Regressor
   * Random Forest Regressor
The models are evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared Score (R²). 
### 2. Dataset
* Source: sklearn.datasets.fetch_california_housing
* Features include:
    - MedInc → Median income
    - HouseAge → Age of the house
    - AveRooms → Average number of rooms
    - AveBedrms → Average number of bedrooms
    - Population → Population in the block
    - AveOccup → Average occupancy
    - Latitude → Geographic latitude
    - Longitude → Geographic longitude
* Target:
  MedHouseVal → Median house value (in hundreds of thousands of dollars)
  
### 3. Metrics Used
| Metric | Description                                                                        | Goal             |
| ------ | ---------------------------------------------------------------------------------- | ---------------- |
| MSE    | Mean Squared Error: average squared difference between predicted and actual values | Lower is better  |
| RMSE   | Root Mean Squared Error: square root of MSE, same unit as target                   | Lower is better  |
| MAE    | Mean Absolute Error: average absolute difference between predicted and actual      | Lower is better  |
| R²     | R-squared Score: proportion of variance explained by the model                     | Higher is better |

### 4. Results
##### R-Squared Score
![R^2 Comparison](https://raw.githubusercontent.com/nananananani/house_price_prediction/refs/heads/main/r2%20result.png)

##### Model Results
![Results Comparison](https://github.com/nananananani/house_price_prediction/blob/main/results.png?raw=true)

⋆.˚—————————————————————————————————————————————————⋆.˚

### 5. Notes
* Features are scaled using StandardScaler for better model performance.
* No missing values exist in the dataset.

### 6. Author
* Name: Janani Hema
* University: SRMIST KTR
* Club Recruitment Task: ACM SIGKDD
* Year: 2




