import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from tabs.patient_demographics import preprocess_data, inpatient_target

#%%
numerical_fields = {
    "BENE_HI_CVRAGE_TOT_MONS": 12,
    "BENE_SMI_CVRAGE_TOT_MONS": 12,
    "BENE_HMO_CVRAGE_TOT_MONS": 0,
    "PLAN_CVRG_MOS_NUM": 10,
    "MEDREIMB_IP": 6090,
    "BENRES_IP": 1068,
    "PPPYMT_IP": 0,
    "MEDREIMB_OP": 0,
    "BENRES_OP": 0,
    "PPPYMT_OP": 0,
    "MEDREIMB_CAR": 1250,
    "BENRES_CAR": 470,
    "PPPYMT_CAR": 30,
    "CLM_PMT_AMT_INP": 6000,
    "NCH_PRMRY_PYR_CLM_PD_AMT_INP": 0.0,
    "CLM_PASS_THRU_PER_DIEM_AMT_INP": 30,
    "NCH_BENE_IP_DDCTBL_AMT_INP": 1068,
    "NCH_BENE_PTA_COINSRNC_LBLTY_AM_INP": 0,
    "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM_INP": 0,
    "CLM_UTLZTN_DAY_CNT_INP": 3,
    "TotalOutpatientVist": 5,
}
#%%
comorbidity_field = {
    "SP_ALZHDMTA": 1,
    "SP_CHF": 1,
    "SP_CHRNKIDN": 1,
    "SP_CNCR": 2,
    "SP_COPD": 2,
    "SP_DEPRESSN": 1,
    "SP_DIABETES": 1,
    "SP_ISCHMCHT": 1,
    "SP_OSTEOPRS": 2,
    "SP_RA_OA": 2,
    "SP_STRKETIA": 2,
}
icd_diagnosis_fields = {
    "ADMTNG_ICD9_DGNS_CD_INP_CAT": "580-629",
    "ICD9_DGNS_CD_1_INP_CAT": "580-629",
    "ICD9_DGNS_CD_2_INP_CAT": "680-709",
    "ICD9_DGNS_CD_3_INP_CAT": "240-279",
    "ICD9_DGNS_CD_4_INP_CAT": "390-459",
    "ICD9_DGNS_CD_5_INP_CAT": "V01-V91",
    "ICD9_DGNS_CD_6_INP_CAT": "290-319",
    "ICD9_DGNS_CD_7_INP_CAT": "240-279",
    "ICD9_DGNS_CD_8_INP_CAT": "320-389",
    "ICD9_DGNS_CD_9_INP_CAT": "290-319",
}

other_dropdown_fields = [
    "BENE_RACE_CD",
    "BENE_SEX_IDENT_CD",
    "BENE_ESRD_IND",
    "PRVDR_NUM_CAT",
    "ICD9_PRCDR_CD_1_INP_CAT",
]
#%%
diagnosid_code_category = (
    inpatient_target[icd_diagnosis_fields.keys()].melt()["value"].unique()
)
#%%
map_comorbity = [
    {"label": "Yes", "value": 1},
    {"label": "No", "value": 2},
]

map_gender = [
    {"label": "Male", "value": 1},
    {"label": "Female", "value": 2},
]

map_race = [
    {"label": "White", "value": 1},
    {"label": "Black", "value": 2},
    {"label": "Others", "value": 3},
    {"label": "Hispanic", "value": 5},
]

map_esrd = [
    {"label": "Yes", "value": "Y"},
    {"label": "No", "value": "0"},
]

map_prvdr_num_cat = [
    {"label": prvdr_num_cat, "value": prvdr_num_cat}
    for prvdr_num_cat in inpatient_target["PRVDR_NUM_CAT"].unique()
]

map_procedure_code = [
    {"label": proc_code, "value": proc_code}
    for proc_code in inpatient_target["ICD9_PRCDR_CD_1_INP_CAT"].unique()
    if not pd.isnull(proc_code)
]

map_diagnosis_code = [
    {"label": diag_code, "value": diag_code}
    for diag_code in diagnosid_code_category
    if diag_code
]
#%%
tab_2_layout = html.Div(
    [
        html.Div(
            [
                html.P(
                    children=["ENTER DESYNPUF_ID"],
                    id="DESYNPUF_ID_LABEL",
                    className="input__heading",
                ),
                dcc.Input(
                    id="DESYNPUF_ID",
                    type="text",
                    placeholder="DESYNPUF_ID",
                    value="Test Patient",
                    debounce=True,
                    className="oper__input",
                ),
            ],
            className="input__container",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            children=[f"ENTER {k}"],
                            id=f"{k}_LABEL",
                            className="input__heading",
                        ),
                        dcc.Input(
                            id=f"{k}",
                            type="number",
                            placeholder=f"{k}",
                            debounce=True,
                            value=v,
                            className="oper__input",
                        ),
                    ],
                    className="input__container",
                )
                for k, v in numerical_fields.items()
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            children=[f"SELECT IF PATIENT HAS {k}"],
                            id=f"{k}_LABEL",
                            className="input__heading",
                        ),
                        dcc.Dropdown(
                            id=f"{k}",
                            options=map_comorbity,
                            value=v,
                            className="reag__select",
                            placeholder=f"Select {k}.",
                        ),
                    ],
                    className="dropdown__container",
                )
                for k, v in comorbidity_field.items()
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            children=["SELECT PATIENT BENE_RACE_CD"],
                            id="BENE_RACE_CD_LABEL",
                            className="input__heading",
                        ),
                        dcc.Dropdown(
                            id="BENE_RACE_CD",
                            options=map_race,
                            value=5,
                            className="reag__select",
                            placeholder="Select BENE_RACE_CD.",
                        ),
                    ],
                    className="dropdown__container",
                ),
                html.Div(
                    [
                        html.P(
                            children=["SELECT PATIENT BENE_SEX_IDENT_CD"],
                            id="BENE_SEX_IDENT_CD_LABEL",
                            className="input__heading",
                        ),
                        dcc.Dropdown(
                            id="BENE_SEX_IDENT_CD",
                            options=map_gender,
                            value=2,
                            className="reag__select",
                            placeholder="Select BENE_SEX_IDENT_CD.",
                        ),
                    ],
                    className="dropdown__container",
                ),
                html.Div(
                    [
                        html.P(
                            children=["SELECT PATIENT BENE_ESRD_IND"],
                            id="BENE_ESRD_IND_LABEL",
                            className="input__heading",
                        ),
                        dcc.Dropdown(
                            id="BENE_ESRD_IND",
                            options=map_esrd,
                            value="0",
                            className="reag__select",
                            placeholder="Select BENE_ESRD_IND.",
                        ),
                    ],
                    className="dropdown__container",
                ),
                html.Div(
                    [
                        html.P(
                            children=["SELECT PATIENT PRVDR_NUM_CAT"],
                            id="PRVDR_NUM_CAT_LABEL",
                            className="input__heading",
                        ),
                        dcc.Dropdown(
                            id="PRVDR_NUM_CAT",
                            options=map_prvdr_num_cat,
                            value="0001-0879",
                            className="reag__select",
                            placeholder="Select PRVDR_NUM_CAT.",
                        ),
                    ],
                    className="dropdown__container",
                ),
                html.Div(
                    [
                        html.P(
                            children=["SELECT PATIENT ICD9_PRCDR_CD_1_INP_CAT"],
                            id="ICD9_PRCDR_CD_1_INP_CAT_LABEL",
                            className="input__heading",
                        ),
                        dcc.Dropdown(
                            id="ICD9_PRCDR_CD_1_INP_CAT",
                            options=map_procedure_code,
                            value="60-64",
                            className="reag__select",
                            placeholder="Select ICD9_PRCDR_CD_1_INP_CAT.",
                        ),
                    ],
                    className="dropdown__container",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    children=[f"SELECT {k} FOR THE PATIENT"],
                                    id=f"{k}_LABEL",
                                    className="input__heading",
                                ),
                                dcc.Dropdown(
                                    id=f"{k}",
                                    options=map_diagnosis_code,
                                    value=v,
                                    className="reag__select",
                                    placeholder=f"Select {k} FOR THE PATIENT.",
                                ),
                            ],
                            className="dropdown__container",
                        )
                        for k, v in icd_diagnosis_fields.items()
                    ]
                ),
                html.Div(
                    [html.Button(id="submit-button", n_clicks=0, children="Submit"),]
                ),
                html.Div(
                    id="out-all-types",
                    children=["This should change"],
                    className="six column",
                ),
            ]
        ),
    ],
)
