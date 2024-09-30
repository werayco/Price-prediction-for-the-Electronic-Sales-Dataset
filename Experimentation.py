# !pip install mlflow shap lightgbm
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as light
from sklearn.svm import SVR
from utils import trainer
import numpy as np
from mlflow import MlflowClient as mlc
import dagshub
import json

uri = "https://dagshub.com/werayco/Price-prediction-for-the-Electronic-Sales-Dataset.mlflow"
dagshub.init(repo_owner='werayco', repo_name='Price-prediction-for-the-Electronic-Sales-Dataset', mlflow=True)

path_to_data = r"CSVFiles/Transformed_Data.csv"

experiment_id_gradient_boosting = mlflow.get_experiment_by_name("Gradient_Boosting").experiment_id
Gradient_params = {
    "alpha": np.arange(0.4,1,0.1),
    "n_estimators": np.arange(40,130,10),
    "learning_rate": np.logspace(-1,-4,num=4,base=10),
}

experiment_id_SVC = mlflow.get_experiment_by_name("SVC").experiment_id
svc_params = {'C': np.arange(0,10,1),
               'epsilon': np.logspace(-1,-4,num=4,base=10), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

experiment_id_random_forest = mlflow.get_experiment_by_name("RandomForest").experiment_id
rand_forest_params = {
    "criterion": ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    "n_estimators": np.arange(50,500,100),
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 3, 4],
}

# reading our clean csv file
dataframe = pd.read_csv(path_to_data).head(2000)

import mlflow.sklearn
feature = dataframe[["PCA1", "PCA2", "PCA3"]]
label = dataframe["Total Price"]

x_train, x_test, y_train, y_test = train_test_split(
    feature, label, random_state=45, test_size=0.3
)

# A list of tuples containing the respective model information
models = [
    ("gradient_boosting", GradientBoostingRegressor(), Gradient_params,experiment_id_gradient_boosting),
    ("RandomForest", RandomForestRegressor(), rand_forest_params,experiment_id_random_forest),
    ("Support_Vector_Machine", SVR(),svc_params,experiment_id_SVC)
]

# mlflow.set_tracking_uri(uri=uri)
for model_name, model, params,experiment_id in models:

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"{model_name}_For_Electronics Sale"
    ):
        # Hyperparameter tunning Section for Gradient Boosting
        random = RandomizedSearchCV(estimator=model, param_distributions=params)
        random.fit(x_train, y_train)

        # Best Model with the best parameter combination
        best_model = random.best_estimator_
        best_params = random.best_params_

        # logging the metrics to mlflow uri
        mlflow.log_params(best_params)

        # logging the best model to mlflow
        mlflow.sklearn.log_model(best_model, artifact_path=f"{model_name}_run")
        score, mae = trainer(best_model=best_model,x_test=x_test,y_test=y_test,x_train=x_train,y_train=y_train)

        metrics = {f"{model_name}_MAE":mae, f"{model_name}_r2 score": score}

        # logging the metric to mlfloq
        mlflow.log_metric("r2_score", score)
        mlflow.log_metrics(metrics=metrics)

# LightGBM Section -->
experiment_id_lightgbm = mlflow.get_experiment_by_name("LightGBM").experiment_id
lightgbm_params = {
    "learning_rate": [0.1, 0.001, 0.0001],
    "num_leaves": [200, 255, 300, 355],
    "objective": ["regression", "l1", "l2"],
    "boosting_type": ["gbdt"],
    "n_jobs": [10, 12, 14, 16],
    "num_trees": [300, 350, 400, 500],
}
with mlflow.start_run(
    experiment_id=experiment_id_lightgbm, run_name="lightGBM_For_Electronics Sale"
):
    model_reg = light.LGBMRegressor(force_col_wise=True)
    train_dataset = light.Dataset(x_train, label=y_train)

    random_light = RandomizedSearchCV(
        estimator=model_reg,
        param_distributions=lightgbm_params,n_iter=10,n_jobs=3
    )

    random_light.fit(x_train, y_train)
    best_model_light = random_light.best_estimator_
    best_params_light = random_light.best_params_

    runs_id = "runs:/<run_id>/<artifact_name"
    model_reg_rui = "models:/<model_name>/<model_version?"

    score, mae = trainer(best_model=best_model,x_test=x_test,y_test=y_test,x_train=x_train,y_train=y_train)

    mlflow.log_metrics({"r2_score":score, "mae":mae})

    mlflow.log_params(best_params_light)
    mlflow.sklearn.log_model(best_model_light, artifact_path="lightgbmModel")
