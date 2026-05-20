import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Ames Housing dataset
df = pd.read_csv("AmesHousing.csv")

# Select important attributes
features = [
    'Lot Area',
    'Gr Liv Area',
    'Overall Qual',
    'Year Built',
    'Garage Area',
    'Total Bsmt SF'
]

# Target variable
target = 'SalePrice'

# Remove rows with missing values in selected columns
df = df.dropna(subset=features + [target])

# Input and output variables
X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

# Train and evaluate each model
for name, model in models.items():
    print("=" * 50)
    print("Model:", name)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    # Print results
    print("Mean Absolute Error (MAE):", round(mae, 2))
    print("Mean Squared Error (MSE):", round(mse, 2))
    print("Root Mean Squared Error (RMSE):", round(rmse, 2))
    print("R² Score:", round(r2, 4))

print("=" * 50)
print("Project completed successfully.")
