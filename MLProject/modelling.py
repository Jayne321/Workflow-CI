import os

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "housing_preprocessing.csv")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file://" + os.path.join(BASE_DIR, "..", "mlruns"))

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Housing_Prices_Retraining_CI")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Training data not found at: {DATA_PATH}")

# Read the local data file from the MLProject folder so the CI runner and local runs both use the same path.
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable autologging so MLflow registers parameters and metrics perfectly
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Cloud_Retrain_Model"):
    # Using lightweight parameters so the automated cloud build runs fast
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    print("Cloud automated retraining completed successfully!")