from fastapi import FastAPI
from pydantic import BaseModel
from utils import pickle_loader
import pandas as pd
minmax_feat = pickle_loader(r"PickleFiles\minmax_for_feature.pkl")
minmax_label = pickle_loader(r"PickleFiles\minmax_for_label.pkl")
pca_feat = pickle_loader(r"PickleFiles\pca_model.pkl")
encodr_shipping = pickle_loader(r"PickleFiles\encoder_for_shipping_type.pkl")
model = pickle_loader(r"C:\Users\LENOVO-PC\Downloads\model (3).pkl")

app = FastAPI()

class data(BaseModel):
    Quantity:float
    Age:float
    ShippingType:str
    Add_on_Total:float
    Rating:float

@app.get("/home/")
async def home():
    return {"data": "How are you?"}

@app.post("/predictions")
async def preds(data_obj:data):
    data01 = data_obj.dict()
    quantity = data01["Quantity"]
    Age = data01["Age"]
    ShippingType = data01["ShippingType"]
    Add_on_Total = data01["Add_on_Total"]
    Rating = data01["Rating"]

    df = pd.DataFrame({"Quantity":[quantity,quantity],"Age":[Age,Age],
                        "Shipping Type":[ShippingType,ShippingType],
                        "Add-on Total":[Add_on_Total,ShippingType],
                        "Rating":[Rating,ShippingType]})

    df["Shipping Type"] = encodr_shipping.transform(df[["Shipping Type"]]) # encoding the shipping cat. feature
    minmaxed_feat_array = minmax_feat.transform(df.head(1)) # now all the features are normalized

    pca_reduc = pca_feat.transform(minmaxed_feat_array) # reducing the dimension
    y_pred = model.predict(pca_reduc)
    un_minmax_label = minmax_label.inverse_transform(y_pred.reshape(-1,1))

    return {"predict":list(un_minmax_label.flatten())}











