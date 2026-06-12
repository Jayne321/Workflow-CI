FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn pandas numpy

# Copy MLProject files into the container working directory
COPY MLProject/ /app/MLProject/

# Define default execution environment parameters
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Trigger the retraining command automatically upon container instantiation
CMD ["mlflow", "run", "MLProject/", "--entry-point", "main", "--env-manager", "local"]