
from shap.explainers import TreeExplainer
import json
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import pickle_loader
import matplotlib.pyplot as plt

model = pickle_loader(r"C:\Users\LENOVO-PC\Downloads\model (3).pkl")
path_to_data = r"Transformed_Data.csv"

dataframe = pd.read_csv(path_to_data).head(2000)
feature = dataframe[["PCA1", "PCA2", "PCA3"]]


label = dataframe["Total Price"]
x_train, x_test, y_train, y_test = train_test_split(
    feature, label, random_state=45, test_size=0.3

)

import shap # used to identity the most important feature

evaluator = TreeExplainer(model=model,)
shap_values = evaluator.shap_values(X=x_train,y=y_train)
print(shap_values.shape)
print(shap_values)
shap.summary_plot(shap_values,x_train)
plt.show() # The Principal Component PCA2 is the most important feature
plt.savefig("shapley_summary_plot.png")


