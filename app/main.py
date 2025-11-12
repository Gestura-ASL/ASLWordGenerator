from typing import Union
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mediapipe_capture import capture_and_save
from model_inference import run_model
import os


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/run_capture")
def run_capture():
    xyz_path = "../asl-signs/train_landmark_files/16069/10042041.parquet"
    out_path = "output/output.parquet"

    if not os.path.exists("output"):
        os.makedirs("output")

    print("opened  camera ... capturing ....")

    parquet_file = capture_and_save(xyz_path, out_path)
    if parquet_file is None:
        return JSONResponse({"error": "No landmarks captured."}, status_code=400)

    output_array = run_model(parquet_file)
    print("Model output:", output_array)

    return {
        "message": "Capture and model run completed successfully",
        "output": output_array
    }