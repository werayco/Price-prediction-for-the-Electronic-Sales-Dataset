import numpy as np
import pickle as pkl
from sklearn.metrics import r2_score,mean_absolute_error

def anomaly_iqr(dataset):
    """
    **This Uses Interquatile range to identify the outliers.**\n
    lower_bound = q1 - 1.5IQR \n
    higher_bound = q3 + 1.5IQR \n
    """
    q1, q3 = np.percentile(dataset, [25, 75])
    interquatile_range = q3 - q1
    lower_bound = q1 - 1.5 * interquatile_range
    higher_bound = q3 + 1.5 * interquatile_range
    return (lower_bound, higher_bound)


def trainer(best_model,x_train,y_train,x_test,y_test):
    best_model.fit(x_train,y_train)
    y_pred = best_model.predict(x_test)
    score = r2_score(y_true=y_test, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    return (score,mae)

def pickle_saver(path: str, model_instance):
    """
    This function saves your model to a pickle file using the Pickle Library
    """
    with open(path, "wb") as path_alias:
        pkl.dump(model_instance, path_alias)

def pickle_loader(path: str):
    """
    This function saves your model to a pickle file using the Pickle Library
    """
    with open(path, "rb") as path_alias:
        model = pkl.load(path_alias)
        return model
    