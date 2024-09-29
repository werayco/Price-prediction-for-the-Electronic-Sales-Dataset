import mlflow
import dagshub
uri = "https://dagshub.com/werayco/Price-prediction-for-the-Electronic-Sales-Dataset.mlflow"

dagshub.init(repo_owner='werayco', repo_name='Price-prediction-for-the-Electronic-Sales-Dataset', mlflow=True)
mlflow.set_tracking_uri(uri=uri)

mlflow.create_experiment(
    "LightGBM",
    artifact_location="Mlflow",
    tags={"env": "dev", "version": "1.0.0"},
)

mlflow.create_experiment(
    "Gradient_Boosting",
    artifact_location="Mlflow",
    tags={"env": "dev", "version": "1.0.0"},
)

mlflow.create_experiment(
    "RandomForest",
    artifact_location="Mlflow",
    tags={"env": "dev", "version": "1.0.0"},
)

mlflow.create_experiment(
    "SVC",
    artifact_location="Mlflow",
    tags={"env": "dev", "version": "1.0.0"},
)
