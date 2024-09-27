import mlflow

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
