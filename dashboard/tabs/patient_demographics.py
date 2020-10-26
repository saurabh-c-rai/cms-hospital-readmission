#%%
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import sqlite3
import plotly.express as px

#%%
import json
import os

#%%
cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)
#%%
with open("../../config.json", "r") as f:
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
conn_object = sqlite3.connect(f"../{DATABASE_PATH}")
#%%
colors = {"background": "#F3F6FA", "background_div": "white", "text": "#7FDBFF"}
#%%
inpatient_target = pd.read_sql("SELECT * FROM InPatient_Target", con=conn_object)
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
        labels={"BENE_RACE_CD": "RACE", "0": "PATIENTS"},
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
                        html.Div(
                            "Text here",
                            style={"textAlign": "center"},
                            className="CommentSection",
                            id="GenderComments",
                        ),
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
                            "Race - Gender distribution", style={"textAlign": "center"}
                        ),
                        dcc.Graph(id="race_gender-graph-4", figure=gender_race_plot(),),
                    ],
                    className="six columns",
                ),
                html.Div(
                    [
                        html.H6(
                            "Age - Gender Disrtibution", style={"textAlign": "center"}
                        ),
                        dcc.Graph(
                            id="example-graph-5",
                            figure={"data": [], "layout": {"title": "Graph-5"}},
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
                            "Race - Age Disrtibution", style={"textAlign": "center"}
                        ),
                        dcc.Graph(
                            id="example-graph-6",
                            figure={"data": [], "layout": {"title": "Graph-6"},},
                        ),
                    ]
                )
            ],
            className="row",
            style={"margin": "1% 3%"},
        ),
    ]
)
