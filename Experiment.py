# !pip install mlflow shap lightgbm
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as light
import shap
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

path_to_data = r"Transformed_Data.csv"

Gradient_params = {
    "alpha": [0.6, 0.7, 0.8, 0.9],
    "n_estimators": [50, 75, 80, 95, 100],
    "max_depth": [2, 3, 5, 8],
    "learning_rate": [0.1, 0.001, 0.0001],
}

rand_forest_params = {
    "min_samples_split": [1,2,3,4],
    "criterion": ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    "n_estimators": [100,120,150,190],
    "max_features": ["sqrt", "log2"],
}

lightgbm_params = {
    "learning_rate": [0.1, 0.001, 0.0001],
    "num_leaves": [200, 255, 300, 355],
    "objective": ["regression", "l1", "l2"],
    "boosting_type": ["gbdt"],
    "n_jobs": [10, 12, 14, 16],
    "num_trees": [300, 350, 400, 500],
}

# Retrieving the experiment ID's for each Experiment
experiment_id_gradient_boosting = mlflow.get_experiment_by_name("Gradient_Boosting").experiment_id
experiment_id_lightgbm = mlflow.get_experiment_by_name("LightGBM").experiment_id
experiment_id_random_forest = mlflow.get_experiment_by_name("RandomForest").experiment_id
experiment_id_SVC = mlflow.get_experiment_by_name("SVC").experiment_id

# reading our clean csv file
dataframe = pd.read_csv(path_to_data)

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
]

for model_name, model, params,experiment_id in models:

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"{model_name}_For_Electronics Sale"
    ):
        # Hyperparameter tunning Section for Gradient Boosting
        random = GridSearchCV(estimator=model, param_grid=Gradient_params)
        random.fit(x_train, y_train)

        # Best Model with the best parameter combination
        best_model = random.best_estimator_
        best_params = random.best_params_

        # logging the metrics to mlflow uri
        mlflow.log_params(best_params)

        # logging the best model to mlflow
        mlflow.sklearn.log_model(best_model, artifact_path=f"{model_name}_run")

        # calculating the ypred
        y_pred = best_model.predict(x_test)
        score = r2_score(y_true=y_test, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)

        metrics = {f"{model_name}_MAE":mae, f"{model_name}_r2 score": score}

        # logging the metric to mlfloq
        mlflow.log_metric("r2_score", score)
        mlflow.log_metrics(metrics=metrics)



with mlflow.start_run(
    experiment_id=experiment_id_lightgbm, run_name="lightGBM_For_Electronics Sale"
):
    model_reg = light.LGBMRegressor(force_col_wise=True)
    train_dataset = light.Dataset(x_train, label=y_train)

    random_light = GridSearchCV(
        estimator=model_reg,
        param_grid=lightgbm_params,
    )

    random_light.fit(x_train, y_train)
    best_model_light = random_light.best_estimator_
    best_params_light = random_light.best_params_

    y_pred_light = random_light.predict(x_test)
    score_r2 = r2_score(y_true=y_test, y_pred=y_pred_light)

    mlflow.log_metric("r2_score", score_r2)
    mlflow.log_params(best_params_light)
    mlflow.sklearn.log_model(best_model_light, artifact_path="lightgbmModel")
