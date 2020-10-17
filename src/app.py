#%%
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import traceback

#%%
import os

cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)
# from flask_restful import reqparse
#%%
import json

with open("../config.json", "r") as f:
    config = json.load(f)
#%%

MODEL_REPOSITORY_LOCATION = config[1]["model_repository_location"]

#%%
app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello():
    return "API Endpoint for CMS READMISSION PROJECT"


@app.route("/predict", methods=["POST"])
def predict():
    rfc_rfe_randover_pipeline = joblib.load(
        f"{MODEL_REPOSITORY_LOCATION}\\rfc_rfe_randover_pipeline.pkl"
    )
    if rfc_rfe_randover_pipeline:
        try:
            request_value = request.get_json()
            model_columns = joblib.load(
                f"{MODEL_REPOSITORY_LOCATION}\\rfc_rfe_randover_pipeline_cols.pkl"
            )
            dataframe = pd.DataFrame(columns=model_columns)
            for query in request_value:
                dataframe = dataframe.append(query, ignore_index=True)

            dataframe = dataframe.applymap(
                lambda x: np.nan if x in ["", "nan", "None", ""] else x
            )
            prediction = rfc_rfe_randover_pipeline.predict(dataframe)
            print("here:", prediction)
            return jsonify({"prediction": str(prediction)})

        except:
            return jsonify({"trace": traceback.format_exc()})
    else:
        return "No model here to use"


if __name__ == "__main__":
    app.run(debug=True)

