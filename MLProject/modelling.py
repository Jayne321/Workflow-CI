import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# WAJIB mengarah ke localhost sesuai instruksi gambar Anda
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Housing_Local_Basic")

# Membaca data bersih hasil preprocessing Kriteria 1
df = pd.read_csv("housing_preprocessing.csv")
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengaktifkan fungsi autolog untuk standar Basic
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Baseline_Model"):
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    print("Pelatihan model lokal (Basic) selesai.")