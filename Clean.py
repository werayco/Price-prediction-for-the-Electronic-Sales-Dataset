import pandas as pd
from datetime import datetime
import os

os.makedirs("PickleFiles", exist_ok=True)
os.makedirs("CSVClean", exist_ok=True)

startime = datetime.now()
import time
from sklearn.decomposition import PCA
from utils import anomaly_iqr
import numpy as np

# Highligting the features and target
features = ["Quantity", "Age", "Shipping Type", "Add-on Total", "Rating"]
target = ["Total Price"]  # Numerical

# Loading the data and dropping the nan records
data = pd.read_csv(r"okayy.csv").dropna().drop_duplicates()
x_features = data[features]
y_label = data[target]  # Total Price is now set to be the label

# Data Cleaning
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# Categorical features
encoder = OrdinalEncoder()  # for 'Shipping Type' Column
minmax = MinMaxScaler()  # for features
imputer = SimpleImputer(strategy="mean")  # for features

shipping = list(
    encoder.fit_transform(
        x_features[
            ["Shipping Type"]
        ]  # 'Shipping Type' Column is now a Numerical Column
    ).flatten()  # turning the 2D array into a 1D array using the flatten method
)

x_features["Shipping Type"] = (
    shipping  # 'Shipping Type' Column is now a Numerical Column
)

import pickle as pkl
import os

def pickle_saver(path: str, model_instance):
    """
    This function saves your model to a pickle file using the Pickle Library
    """
    with open(path, "wb") as path_alias:
        pkl.dump(model_instance, path_alias)

pickle_saver(path="PickleFiles/encoder_for_shipping_type.pkl", model_instance=encoder)

for col in x_features.select_dtypes(
    include=[np.number]
).columns:  # Removing the outliers from the 'Quantity", "Age", "Add-on Total", "Rating' Columns
    low, high = anomaly_iqr(dataset=x_features[col])
    print(f"{col}: [{low,high}]")
    x_features = x_features[(x_features[col] >= low) & (x_features[col] <= high)]


for col in y_label.select_dtypes(
    include=[np.number]
).columns:  # Removing the outliers from the target variable
    low, high = anomaly_iqr(dataset=y_label[col])
    print(f"{col}: [{low,high}]")
    y_label = y_label[(y_label[col] >= low) & (y_label[col] <= high)]

row, _ = x_features.shape
x_features, y_label = x_features.dropna(), y_label.dropna()
y_label = y_label.head(row)  #  same row with feature

# saving this un-normalized data to csv for future referencing
dataframe1 = pd.concat([x_features, y_label], axis=1).dropna()
dataframe1.to_csv("Data.csv", index=False)

from sklearn.preprocessing import MinMaxScaler

minmax_label = MinMaxScaler()
minmaxer_label = minmax_label.fit_transform(
    y_label
)  # this is the normalized label (array)
pickle_saver(path="PickleFiles/minmax_for_label.pkl", model_instance=minmax_label)

# Feature selection/ reduction using pca
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

minmax = MinMaxScaler()
minmaxer = minmax.fit_transform(x_features)  # this is the normalized feature (array)
pickle_saver(path="PickleFiles/minmax_for_feature.pkl", model_instance=minmax)

pca = PCA(n_components=3)
pca_reduct = pca.fit_transform(X=minmaxer)

pickle_saver(path="PickleFiles/pca_model.pkl", model_instance=pca)

label = pd.DataFrame(minmaxer_label, columns=["Total Price"])
feat = pd.DataFrame(pca_reduct, columns=["PCA1", "PCA2", "PCA3"])

dataframe = pd.concat([feat, label], axis=1)
dataframe.to_csv("CSVClean/Transformed_Data.csv", index=False)

print(x_features.shape, y_label.shape)
