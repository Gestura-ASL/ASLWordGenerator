from typing import Union
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mediapipe_capture import capture_and_save
from model_inference import run_model
import os
import requests


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


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

"""
update remote_url with the actual backend api 

"""
@app.get("/send_output")
def send_output():
    global latest_output

    if latest_output is None:
        return JSONResponse({"error": "No model output found. Run /run_capture first."}, status_code=400)



    remote_url = " "

    try:
        response = requests.post(remote_url, json={"output": latest_output})
        response.raise_for_status()
        print("Sent to remote backend successfully.")
    except requests.RequestException as e:
        print(" Error sending to backend:", e)
        return JSONResponse(
            {"message": "Failed to send output to backend", "error": str(e)},
            status_code=500
        )

    return {
        "message": "Output sent successfully",
        "remote_response": response.json() if response.content else None
    }