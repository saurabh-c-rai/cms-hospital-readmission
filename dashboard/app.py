#%%
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
from dashboard.tabs import (  # pylint: disable=import-error
    patient_demographics,
    patient_med_readmit,
    readmission_plots,
)
import plotly.figure_factory as ff
import numpy as np
import requests
import plotly.graph_objs as go
import json
import dash_daq as daq

#%%
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

colors = {"background": "#F3F6FA", "background_div": "white", "text": "black"}

app.config["suppress_callback_exceptions"] = True
#%%
URL = "http://127.0.0.1:5000/predict"
HEADER = {"Content-Type": "application/json"}

ALLOWED_FIELDS = (
    ["DESYNPUF_ID"]
    + list(patient_med_readmit.numerical_fields.keys())
    + list(patient_med_readmit.comorbidity_field.keys())
    + list(patient_med_readmit.icd_diagnosis_fields.keys())
    + patient_med_readmit.other_dropdown_fields
)
#%%
app.layout = html.Div(
    style={"backgroundColor": colors["background"], "color": colors["text"]},
    children=[
        html.H1("Patient Re-admissions Dashboard", style={"textAlign": "center",},),
        daq.ToggleSwitch(  # pylint: disable=not-callable
            id="Toggle",
            value=True,
            label={"label": "Include Readmission",},
            labelPosition="bottom",
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

#%%
@app.callback(
    Output("tabs-content", "children"),
    [Input("tabs", "value"), Input("Toggle", "value"),],
)
def render_content(tab, toggle_value):
    if tab == "dem_tab":
        if toggle_value:
            return patient_demographics.tab_1_layout
        else:
            return readmission_plots.tab_1_layout
    elif tab == "med_tab":
        return patient_med_readmit.tab_2_layout


#%%
@app.callback(
    Output("out-all-types", "children"),
    [Input("submit-button", "n_clicks")],
    [State(f"{_}", "value") for _ in ALLOWED_FIELDS],
)
def get_prediction_from_model(n_clicks, *args):
    if n_clicks >= 1:
        payload = dict(zip(ALLOWED_FIELDS, [arg for arg in (args or [])]))
        response = requests.request(
            "POST", URL, headers=HEADER, data=json.dumps([payload])
        )
        return str(response.text.encode("utf8"))


#%%
if __name__ == "__main__":
    app.run_server(debug=True)

