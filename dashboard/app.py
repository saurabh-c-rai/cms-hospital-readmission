import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
from tabs import patient_demographics, patient_med_readmit
import pandas as pd
import plotly.figure_factory as ff
import numpy as np
import requests
import plotly.graph_objs as go

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

colors = {"background": "#F3F6FA", "background_div": "white", "text": "#009999"}

app.config["suppress_callback_exceptions"] = True

URL = "http://127.0.0.1:5000/predict"
HEADER = {"Content-Type": "application/json"}

ALLOWED_FIELDS = (
    ["DESYNPUF_ID"]
    + patient_med_readmit.numerical_fields
    + patient_med_readmit.comorbidity_field
    + patient_med_readmit.icd_diagnosis_fields
    + patient_med_readmit.other_dropdown_fields
)

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            "Patient Re-admissions Dashboard",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        dcc.Tabs(
            id="tabs",
            className="row",
            style={"margin": "2% 3%", "height": "20", "verticalAlign": "middle"},
            value="dem_tab",
            children=[
                dcc.Tab(label="Demographics", value="dem_tab"),
                dcc.Tab(label="Medical Speciality And Re-Admission", value="med_tab")
                # dcc.Tab(label='Re-admissions', value='readmit_tab')
            ],
        ),
        html.Div(id="tabs-content"),
    ],
)


@app.callback(Output("tabs-content", "children"), [Input("tabs", "value")])
def render_content(tab):
    if tab == "dem_tab":
        return patient_demographics.tab_1_layout
    elif tab == "med_tab":
        return patient_med_readmit.tab_2_layout


@app.callback(
    Output("out-all-types", "children"),
    [Input("submit-button", "n_clicks")],
    [State(f"{_}", "value") for _ in ALLOWED_FIELDS],
)
def check_output(n_clicks, *args):
    payload = list(dict(zip(ALLOWED_FIELDS, [arg for arg in (args or [])])))
    response = requests.request("POST", URL, headers=HEADER, data=payload)

    return response


if __name__ == "__main__":
    app.run_server(debug=True)

