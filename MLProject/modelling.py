import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "housing_preprocessing.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Training data not found at: {DATA_PATH}")

# Read the local data file from the MLProject folder
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable autologging so MLflow registers parameters and metrics perfectly
mlflow.sklearn.autolog()

# No start_run block needed here! 
# The MLflow runner automatically handles the active run tracking environment.
print("Starting training script execution inside the cloud environment...")
model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("Cloud automated retraining completed successfully!")