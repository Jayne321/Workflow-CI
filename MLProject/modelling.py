import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

mlflow.set_experiment("Housing_Prices_Retraining_CI")

df = pd.read_csv("housing_preprocessing.csv")
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