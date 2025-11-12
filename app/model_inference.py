import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json

ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def run_model(pqs_file):
    # load the .tflite model
    interpreter = tf.lite.Interpreter(model_path="../models/model.tflite")
    # get the list of available signatures
    found_signatures = list(interpreter.get_signature_list().keys())
    # get the callable for the default signature
    prediction_fn = interpreter.get_signature_runner("serving_default")

    train = pd.read_csv('../asl-signs/train.csv')
    train['sign_ord'] = train['sign'].astype('category').cat.codes

    # Dictionaries to translate sign <-> ordinal encoded sign

    SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

    pq_file = pqs_file
    xyz_np = load_relevant_data_subset(pq_file)

    prediction = prediction_fn(inputs=xyz_np)

    # take it as signs
    sign = prediction['outputs']

    TOP_K = 10
    top_indices = np.argsort(sign)[::-1][:TOP_K]

    payload = [
        {"sign": ORD2SIGN[i], "conf": round(float(sign[i]), 4), "idx": int(i)}
        for i in top_indices
    ]


    json_payload = json.dumps(payload)
    return json_payload



