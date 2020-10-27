#%%
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import sqlite3
import plotly.express as px
from utils.CategorizeCardinalData import (
    ProviderNumCategoryCreator,
    ProcedureCodeCategoryCreator,
    DiagnosisCodeCategoryCreator,
)

#%%
import json
import os

#%%
with open("../config.json", "r") as f:
    config = json.load(f)

#%%
# configuration constant variables
INFREQUENT_CATEGORY_CUT_OFF = config[3]["INFREQUENT_CATEGORY_CUT_OFF"]
INFREQUENT_CATEGORY_LABEL = config[3]["INFREQUENT_CATEGORY_LABEL"]
RANDOM_STATE = config[3]["RANDOM_STATE"]
MISSING_VALUE_LABEL = config[3]["MISSING_VALUE_LABEL"]
EDA_REPORT_LOCATION = config[1]["eda_report_location"]
MODEL_REPOSITORY_LOCATION = config[1]["model_repository_location"]
NON_NAN_THRESHOLD = config[0]["threshold_nan"]
TEST_SIZE_FOR_SPLIT = config[3]["TEST_SIZE_FOR_SPLIT"]
RANDOM_STATE = config[3]["RANDOM_STATE"]
N_JOB_PARAM_VALUE = config[3]["N_JOB_PARAM_VALUE"]
VERBOSE_PARAM_VALUE = config[3]["VERBOSE_PARAM_VALUE"]
DATABASE_PATH = config[1]["database_path"]
#%%
conn_object = sqlite3.connect(f"{DATABASE_PATH}")
#%%
colors = {"background": "#F3F6FA", "background_div": "white", "text": "#7FDBFF"}
#%%
prvdr_num = ProviderNumCategoryCreator()
proc_code = ProcedureCodeCategoryCreator()
diag_code = DiagnosisCodeCategoryCreator()

#%%
def preprocess_data(dataframe: pd.DataFrame):
    # get diagnosis, procedure and hcpcs column list
    diagnosis_code = [col for col in dataframe.columns if "ICD9_DGNS" in col]
    icd_procedural_code = [col for col in dataframe.columns if "ICD9_PRCDR" in col]
    icd_hcpcs_code = [col for col in dataframe.columns if "HCPCS_CD" in col]

    # process prvdr_num cols
    prvdr_num.get_categories_for_providers(dataframe["PRVDR_NUM_INP"])
    dataframe = dataframe.merge(
        right=prvdr_num.unique_prvdr_num_category_df,
        left_on=["PRVDR_NUM_INP"],
        right_on=["PRVDR_NUM"],
        how="left",
    )

    # convert all columns to string
    dataframe[diagnosis_code + icd_procedural_code + icd_hcpcs_code] = (
        dataframe[diagnosis_code + icd_procedural_code + icd_hcpcs_code]
        .astype("str")
        .replace(["nan", "na"], np.nan)
    )

    # process procedural code cols
    proc_code.get_categories_for_procedure_code(dataframe[icd_procedural_code])
    for col in icd_procedural_code:
        dataframe[f"{col}"] = dataframe[f"{col}"].str[:2]
        dataframe[f"{col}_CAT"] = pd.merge(
            left=dataframe,
            right=proc_code.unique_procedure_code_category_df,
            left_on=col,
            right_on="Procedure_code",
            how="left",
        )["Procedure_code_CAT"]

    # processing of diagnosis code
    diag_code.get_categories_for_diagnosis_code(dataframe[diagnosis_code])
    for col in diagnosis_code:
        dataframe[f"{col}"] = dataframe[f"{col}"].str[:3]
        dataframe[f"{col}_CAT"] = pd.merge(
            left=dataframe,
            right=diag_code.unique_diagnosis_code_category_df,
            left_on=col,
            right_on="Diagnosis_code",
            how="left",
        )["Diagnosis_code_CAT"]

    # drop duplicate columns
    dataframe.drop(
        columns=icd_procedural_code + icd_hcpcs_code + diagnosis_code, inplace=True
    )
    # return dataframe
    return dataframe


#%%
inpatient_target = pd.read_sql("SELECT * FROM InPatient_Target", con=conn_object)
inpatient_target = preprocess_data(inpatient_target)

#%%
def gender_pie_chart():
    data = inpatient_target.drop_duplicates(
        subset=["DESYNPUF_ID", "BENE_SEX_IDENT_CD"]
    )["BENE_SEX_IDENT_CD"].value_counts()

    fig = px.pie(data_frame=data, names=data.index, values=data.values)
    return fig


def race_pie_chart():
    data = inpatient_target.drop_duplicates(subset=["DESYNPUF_ID", "BENE_RACE_CD"])[
        "BENE_RACE_CD"
    ].value_counts()

    fig = px.pie(data_frame=data, names=data.index, values=data.values)
    return fig


def age_barplot():
    labels = ["25-40", "40-55", "55-70", "70-85", "85-100"]
    data = inpatient_target.drop_duplicates(subset=["DESYNPUF_ID", "Age"], keep="last")
    age_data_bins = pd.cut(data["Age"], bins=5, labels=labels).value_counts()
    fig = px.bar(
        data_frame=age_data_bins,
        x=age_data_bins.index,
        y=age_data_bins.values,
        labels={"index": "Age Bins", "y": "Count"},
    )
    return fig


def gender_race_plot():
    """
    docstring
    """
    data = (
        inpatient_target.groupby(["BENE_RACE_CD", "BENE_SEX_IDENT_CD"])
        .size()
        .reset_index()
    )

    figure = px.bar(
        data_frame=data,
        x="BENE_RACE_CD",
        y=0,
        color="BENE_SEX_IDENT_CD",
        labels={
            "BENE_RACE_CD": "RACE",
            "0": "PATIENTS",
            "BENE_SEX_IDENT_CD": "PATIENT GENDER",
        },
    )
    return figure


def race_age_plot():
    """
    docstring
    """
    labels = ["25-40", "40-55", "55-70", "70-85", "85-100"]
    inpatient_target["Age_Category"] = pd.cut(
        inpatient_target["Age"], bins=5, labels=labels
    )
    data = (
        inpatient_target.groupby(["Age_Category", "BENE_RACE_CD"]).size().reset_index()
    )
    figure = px.bar(
        data_frame=data,
        x="Age_Category",
        y=0,
        color="BENE_RACE_CD",
        labels={
            "Age_Category": "Age Bins",
            "0": "PATIENTS",
            "BENE_RACE_CD": "PATIENT RACE",
        },
    )
    figure.update_layout(
        # barmode="group",
        title={"text": "Distribution of Race vs Age", "y": 0.9, "x": 0.5,},
    )

    return figure


def gender_age_plot():
    """
    docstring
    """
    labels = ["25-40", "40-55", "55-70", "70-85", "85-100"]
    inpatient_target["Age_Category"] = pd.cut(
        inpatient_target["Age"], bins=5, labels=labels
    )
    data = (
        inpatient_target.groupby(["Age_Category", "BENE_SEX_IDENT_CD"])
        .size()
        .reset_index()
    )
    figure = px.bar(
        data_frame=data,
        x="Age_Category",
        y=0,
        color="BENE_SEX_IDENT_CD",
        labels={
            "Age_Category": "Age Bins",
            "0": "PATIENTS",
            "BENE_SEX_IDENT_CD": "PATIENT GENDER",
        },
    )
    figure.update_layout(
        barmode="group",
        title={"text": "Distribution of Gender vs Age", "y": 0.9, "x": 0.5,},
    )

    return figure


def comorbidity_plot():
    """
    docstring
    """
    comorbidity_col_list = [
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
        "BENE_ESRD_IND",
    ]
    map_dict = {1: "Yes", 2: "No", "Y": "Yes", "0": "No"}
    comorbidity_df = pd.DataFrame()
    for col in comorbidity_col_list:
        comorbidity_df[col] = inpatient_target[col].map(map_dict)

    data = (
        comorbidity_df[comorbidity_col_list].apply(pd.Series.value_counts).reset_index()
    )
    data = data.melt(id_vars="index")
    figure = px.bar(
        data_frame=data,
        x="variable",
        y="value",
        color="index",
        text="value",
        labels={
            "variable": "Co-morbidity Name",
            "value": "PATIENTS COUNT",
            "index": "Presence",
        },
    )
    figure.update_traces(textposition="outside",)
    figure.update_layout(
        barmode="group",
        title={"text": "Co-morbidity in population", "y": 0.9, "x": 0.5,},
    )
    return figure


def prvdr_num_plot():
    """
    docstring
    """
    data = (
        inpatient_target[["DESYNPUF_ID", "PRVDR_NUM_CAT"]]
        .drop_duplicates()
        .groupby(["PRVDR_NUM_CAT"])["DESYNPUF_ID"]
        .size()
        .reset_index()
    )
    figure = px.bar(
        data_frame=data,
        x="PRVDR_NUM_CAT",
        y="DESYNPUF_ID",
        text="DESYNPUF_ID",
        labels={"PRVDR_NUM_CAT": "PROVIDER TYPES", "DESYNPUF_ID": "PATIENTS COUNT",},
    )
    figure.update_traces(textposition="outside",)
    figure.update_layout(
        barmode="group",
        title={
            "text": "Population visiting different Provider types",
            "y": 0.9,
            "x": 0.5,
        },
    )
    return figure


#%%
tab_1_layout = html.Div(
    [
        # html.H3('Patient Demographics'),
        html.Div(
            [
                html.Div(
                    [
                        html.H6("Gender", style={"textAlign": "center"}),
                        dcc.Graph(id="gender-graph-1", figure=gender_pie_chart(),),
                    ],
                    className="four columns",
                ),
                html.Div(
                    [
                        html.H6("Race", style={"textAlign": "center"}),
                        dcc.Graph(id="race-graph-2", figure=race_pie_chart()),
                    ],
                    className="four columns",
                ),
                html.Div(
                    [
                        html.H6("Age", style={"textAlign": "center"}),
                        dcc.Graph(id="age-graph-3", figure=age_barplot(),),
                    ],
                    className="four columns",
                ),
            ],
            className="row",
            style={"margin": "1% 3%"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H6(
                            "Race - Gender Distribution", style={"textAlign": "center"}
                        ),
                        dcc.Graph(id="race_gender-graph-4", figure=gender_race_plot(),),
                    ],
                    className="four columns",
                ),
                html.Div(
                    [
                        html.H6(
                            "Age - Gender Distribution", style={"textAlign": "center"}
                        ),
                        dcc.Graph(id="age-gender-graph-5", figure=gender_age_plot(),),
                    ],
                    className="four columns",
                ),
                html.Div(
                    [
                        html.H6(
                            "Race - Age Distribution", style={"textAlign": "center"},
                        ),
                        dcc.Graph(id="age-race-graph-6", figure=race_age_plot(),),
                    ],
                    className="four columns",
                ),
            ],
            className="row",
            style={"margin": "1% 3%"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H6(
                            "Patient Comorbidity Distribution",
                            style={"textAlign": "center"},
                        ),
                        dcc.Graph(id="comorbidity-graph-7", figure=comorbidity_plot(),),
                    ],
                    className="six columns",
                ),
                html.Div(
                    [
                        html.H6(
                            "Provider Num Distribution", style={"textAlign": "center"}
                        ),
                        dcc.Graph(id="prvdr_num-graph-5", figure=prvdr_num_plot(),),
                    ],
                    className="six columns",
                ),
                # html.Div(
                #     [
                #         html.H6(
                #             "Race - Age Distribution", style={"textAlign": "center"},
                #         ),
                #         dcc.Graph(id="age-race-graph-6", figure=race_age_plot(),),
                #     ],
                #     className="four columns",
                # ),
            ],
            className="row",
            style={"margin": "1% 3%"},
        ),
    ]
)
