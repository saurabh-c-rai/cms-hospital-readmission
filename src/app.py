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
    rfc_pipeline = joblib.load(f"{MODEL_REPOSITORY_LOCATION}\\rfc_pipeline.pkl")
    if rfc_pipeline:
        try:
            request_value = request.get_json()
            print(request_value)
            model_columns = joblib.load(
                f"{MODEL_REPOSITORY_LOCATION}\\model_columns.pkl"
            )
            dataframe = pd.DataFrame(columns=model_columns)
            for query in request_value:
                dataframe = dataframe.append(query, ignore_index=True)

            dataframe = dataframe.applymap(
                lambda x: np.nan if x in ["", "nan", "None", ""] else x
            )
            categorical_features = [
                "BENE_SEX_IDENT_CD",
                "BENE_RACE_CD",
                "BENE_ESRD_IND",
                "SP_ALZHDMTA",
                "SP_CHF",
                "SP_CHRNKIDN",
                "SP_CNCR",
                "SP_COPD",
                "SP_DEPRESSN",
                "SP_DIABETES",
                "SP_ISCHMCHT",
                "SP_OSTEOPRS",
                "SP_RA_OA",
                "SP_STRKETIA",
                "BENE_STATE_COUNTY_CODE",
                "PRVDR_NUM_CAT_INP",
                "ADMTNG_ICD9_DGNS_CD_CAT_INP",
                "ICD9_DGNS_CD_1_CAT_INP",
                "ICD9_DGNS_CD_2_CAT_INP",
                "ICD9_DGNS_CD_3_CAT_INP",
                "ICD9_DGNS_CD_4_CAT_INP",
                "ICD9_DGNS_CD_5_CAT_INP",
                "ICD9_DGNS_CD_6_CAT_INP",
                "ICD9_DGNS_CD_7_CAT_INP",
                "ICD9_DGNS_CD_8_CAT_INP",
                "ICD9_DGNS_CD_9_CAT_INP",
                "ICD9_PRCDR_CD_1_CAT_INP",
            ]
            dataframe[categorical_features] = dataframe[categorical_features].astype(
                "category"
            )
            prediction = rfc_pipeline.predict_proba(dataframe)
            pos_prediction = prediction[:, 1]
            final_predicion = (pos_prediction > 0.364).astype("int")
            final_predicion = list(
                zip(dataframe["DESYNPUF_ID"].tolist(), final_predicion)
            )
            print(
                "here:", list(zip(dataframe["DESYNPUF_ID"].tolist(), final_predicion))
            )
            return jsonify({"prediction": str(final_predicion)})

        except:
            return jsonify({"trace": traceback.format_exc()})
    else:
        return "No model here to use"


if __name__ == "__main__":
    app.run(debug=True)

