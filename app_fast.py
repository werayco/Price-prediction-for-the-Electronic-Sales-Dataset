from fastapi import FastAPI, Response, requests

app = FastAPI()


@app.get("/home/<int:id>")
def home(requests, id):
    return {"data": 344}
