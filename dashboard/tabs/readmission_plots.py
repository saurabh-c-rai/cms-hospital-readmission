#%%
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import sqlite3
import plotly.express as px
from utils.CategorizeCardinalData import (
    ProviderNumCategoryCreator,
    ProcedureCodeCategoryCreator,
    DiagnosisCodeCategoryCreator,
)
from plotly.subplots import make_subplots

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
continous_cols = [
    "BENE_HI_CVRAGE_TOT_MONS",
    "BENE_SMI_CVRAGE_TOT_MONS",
    "BENE_HMO_CVRAGE_TOT_MONS",
    "PLAN_CVRG_MOS_NUM",
    "MEDREIMB_IP",
    "BENRES_IP",
    "PPPYMT_IP",
    "MEDREIMB_OP",
    "BENRES_OP",
    "PPPYMT_OP",
    "MEDREIMB_CAR",
    "BENRES_CAR",
    "PPPYMT_CAR",
    "NCH_PRMRY_PYR_CLM_PD_AMT_INP",
    "CLM_PASS_THRU_PER_DIEM_AMT_INP",
    "NCH_BENE_IP_DDCTBL_AMT_INP",
    "NCH_BENE_PTA_COINSRNC_LBLTY_AM_INP",
    "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM_INP",
    "TotalOutpatientVist",
    "CLM_UTLZTN_DAY_CNT_INP",
]
#%%
def gender_readmission_bar_chart():
    data = (
        inpatient_target.drop_duplicates(subset=["BENE_SEX_IDENT_CD", "DESYNPUF_ID"])
        .groupby(["BENE_SEX_IDENT_CD", "IsReadmitted"])["DESYNPUF_ID"]
        .count()
        .reset_index()
    )
    fig = px.bar(
        data_frame=data,
        x="BENE_SEX_IDENT_CD",
        y="DESYNPUF_ID",
        facet_col="IsReadmitted",
        text="DESYNPUF_ID",
        color="BENE_SEX_IDENT_CD",
        labels={"BENE_SEX_IDENT_CD": "PATIENT GENDER", "DESYNPUF_ID": "PATIENT COUNT"},
    )
    fig.update_traces(textposition="outside",)
    return fig


def continuous_col_box_plot():
    """
    docstring
    """
    figure = px.box(
        data_frame=inpatient_target[continous_cols + ["IsReadmitted"]],
        facet_col="IsReadmitted",
        labels={"variable": "Numerical Features"},
    )
    figure.update_layout(yaxis=dict(domain=[0.5, 0.5]))
    return figure


def race_readmission_bar_chart():
    data = (
        inpatient_target.drop_duplicates(subset=["BENE_RACE_CD", "DESYNPUF_ID"])
        .groupby(["BENE_RACE_CD", "IsReadmitted"])["DESYNPUF_ID"]
        .count()
        .reset_index()
    )
    fig = px.bar(
        data_frame=data,
        x="BENE_RACE_CD",
        y="DESYNPUF_ID",
        facet_col="IsReadmitted",
        text="DESYNPUF_ID",
        color="BENE_RACE_CD",
        labels={"BENE_RACE_CD": "PATIENT GENDER", "DESYNPUF_ID": "PATIENT COUNT"},
    )
    fig.update_traces(textposition="outside",)
    return fig


def continuous_cols_age_scatterplot(separator):
    data = inpatient_target.drop_duplicates(subset=["DESYNPUF_ID", "Age"], keep="last")
    figure = px.scatter(
        data_frame=data,
        x="Age",
        y=separator,
        color="BENE_SEX_IDENT_CD",
        facet_col="IsReadmitted",
    )
    figure.update_yaxes(automargin=True)
    return figure


def comorbidity_plot(isReadmitted: bool):
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
    comorbidity_df["Target"] = inpatient_target["IsReadmitted"]

    data = (
        comorbidity_df[comorbidity_df["Target"] == isReadmitted]
        .apply(pd.Series.value_counts)
        .reset_index()
    )
    data = data.iloc[:-1, :-1]
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
        title={
            "text": "Co-morbidity in population With Readmitted = 1"
            if isReadmitted
            else "Co-morbidity in population With Readmitted = 0",
            "y": 0.95,
            "x": 0.5,
        },
    )
    return figure


def prvdr_num_plot():
    """
    docstring
    """
    data = (
        inpatient_target[["DESYNPUF_ID", "PRVDR_NUM_CAT", "IsReadmitted"]]
        .drop_duplicates()
        .groupby(["PRVDR_NUM_CAT", "IsReadmitted"])["DESYNPUF_ID"]
        .size()
        .reset_index()
    )
    figure = px.bar(
        data_frame=data,
        x="PRVDR_NUM_CAT",
        y="DESYNPUF_ID",
        text="DESYNPUF_ID",
        facet_col="IsReadmitted",
        labels={"PRVDR_NUM_CAT": "PROVIDER TYPES", "DESYNPUF_ID": "PATIENTS COUNT",},
    )
    figure.update_traces(textposition="outside",)
    figure.update_layout(
        barmode="group",
        title={
            "text": "Population visiting different Provider types",
            "y": 0.95,
            "x": 0.5,
        },
    )
    return figure


def procedure_code_plot():
    """
    docstring
    """
    data = (
        inpatient_target[["DESYNPUF_ID", "ICD9_PRCDR_CD_1_INP_CAT", "IsReadmitted"]]
        .drop_duplicates()
        .groupby(["ICD9_PRCDR_CD_1_INP_CAT", "IsReadmitted"])["DESYNPUF_ID"]
        .size()
        .reset_index()
    )
    figure = px.bar(
        data_frame=data,
        x="ICD9_PRCDR_CD_1_INP_CAT",
        y="DESYNPUF_ID",
        text="DESYNPUF_ID",
        labels={
            "ICD9_PRCDR_CD_1_INP_CAT": "PROCEDURE GROUPS",
            "DESYNPUF_ID": "PATIENTS COUNT",
        },
        facet_col="IsReadmitted",
    )
    figure.update_traces(textposition="outside",)
    figure.update_layout(
        barmode="group",
        title={"text": "PROCEDURE PERFORMED ON PATIENT", "y": 0.95, "x": 0.5,},
    )
    return figure


def diagnosis_code_plot(isReadmitted: bool):
    """
    docstring
    """
    diagnosis_category = [
        col for col in inpatient_target if ("DGNS" in col) & ("CAT" in col)
    ]
    figure = make_subplots(
        rows=5,
        cols=2,
        subplot_titles=diagnosis_category,
        row_heights=[250, 250, 250, 250, 250],
    )

    for index, col in enumerate(diagnosis_category):
        data = (
            inpatient_target.loc[
                inpatient_target["IsReadmitted"] == isReadmitted, ["DESYNPUF_ID", col]
            ]
            .drop_duplicates()
            .groupby(col)["DESYNPUF_ID"]
            .size()
            .reset_index()
        )
        figure.add_trace(
            go.Bar(
                x=data[col],
                y=data["DESYNPUF_ID"],
                name=col,
                text=data["DESYNPUF_ID"],
                textposition="outside",
                # height=200,
            ),
            row=(index // 2) + 1,
            col=(index % 2) + 1,
        )
    figure.update_layout(
        height=1200,
        title_text="Diagnosis Code vs Patient Count With Readmitted = 1"
        if isReadmitted
        else "Diagnosis Code vs Patient Count With Readmitted = 0",
    )
    return figure


#%%
tab_1_layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H6("Gender", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="gender-graph-1", figure=gender_readmission_bar_chart(),
                        ),
                    ],
                    className="six columns",
                ),
                html.Div(
                    [
                        html.H6("Race", style={"textAlign": "center"}),
                        dcc.Graph(
                            id="race-graph-2", figure=race_readmission_bar_chart()
                        ),
                    ],
                    className="six columns",
                ),
            ],
            className="row",
            style={"margin": "1% 3%"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H6(f"Age vs {col}", style={"textAlign": "center"}),
                        dcc.Graph(
                            id=f"age-{col}-graph-{str(index)}",
                            figure=continuous_cols_age_scatterplot(col),
                        ),
                    ],
                    className="row",
                )
                for index, col in enumerate(continous_cols)
            ],
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H6(
                            "Patient Comorbidity Distribution",
                            style={"textAlign": "center"},
                        ),
                        dcc.Graph(
                            id="comorbidity-no-readmitted-graph-7",
                            figure=comorbidity_plot(False),
                        ),
                    ],
                    className="six columns",
                ),
                html.Div(
                    [
                        html.H6(
                            "Patient Comorbidity Distribution",
                            style={"textAlign": "center"},
                        ),
                        dcc.Graph(
                            id="comorbidity-readmitted-graph-7",
                            figure=comorbidity_plot(True),
                        ),
                    ],
                    className="six columns",
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
                            "Procedure Performed on Patient",
                            style={"textAlign": "center"},
                        ),
                        dcc.Graph(
                            id="procedure_code-graph-9", figure=procedure_code_plot(),
                        ),
                    ],
                    className="six columns",
                ),
                html.Div(
                    [
                        html.H6(
                            "Provider Num Distribution", style={"textAlign": "center"}
                        ),
                        dcc.Graph(id="prvdr_num-graph-8", figure=prvdr_num_plot(),),
                    ],
                    className="six columns",
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
                            "Box Plot Distribution of Continuous Variables",
                            style={"textAlign": "center"},
                        ),
                        dcc.Graph(
                            id="continuous_cols-box-graph-11",
                            figure=continuous_col_box_plot(),
                        ),
                    ],
                    className="twelve columns",
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
                            "Disease Category for Patient",
                            style={"textAlign": "center"},
                        ),
                        dcc.Graph(
                            id="diagnosis_code-readmitted-graph-10",
                            figure=diagnosis_code_plot(True),
                        ),
                    ],
                    className="twelve columns",
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
                            "Disease Category for Patient",
                            style={"textAlign": "center"},
                        ),
                        dcc.Graph(
                            id="diagnosis_code-not-readmitted-graph-11",
                            figure=diagnosis_code_plot(False),
                        ),
                    ],
                    className="twelve columns",
                ),
            ],
            className="row",
            style={"margin": "1% 3%"},
        ),
    ]
)
