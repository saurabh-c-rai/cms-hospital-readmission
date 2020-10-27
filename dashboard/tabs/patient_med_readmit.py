import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from tabs.patient_demographics import preprocess_data, inpatient_target

#%%
numerical_fields = [
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
    "CLM_PMT_AMT_INP",
    "NCH_PRMRY_PYR_CLM_PD_AMT_INP",
    "CLM_ADMSN_DT_INP",
    "CLM_PASS_THRU_PER_DIEM_AMT_INP",
    "NCH_BENE_IP_DDCTBL_AMT_INP",
    "NCH_BENE_PTA_COINSRNC_LBLTY_AM_INP",
    "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM_INP",
    "CLM_UTLZTN_DAY_CNT_INP",
    "NCH_BENE_DSCHRG_DT_INP",
    "TotalOutpatientVist",
]
#%%
comorbidity_field = [
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
]
icd_diagnosis_fields = [
    "ADMTNG_ICD9_DGNS_CD_INP_CAT",
    "ICD9_DGNS_CD_1_INP_CAT",
    "ICD9_DGNS_CD_2_INP_CAT",
    "ICD9_DGNS_CD_3_INP_CAT",
    "ICD9_DGNS_CD_4_INP_CAT",
    "ICD9_DGNS_CD_5_INP_CAT",
    "ICD9_DGNS_CD_6_INP_CAT",
    "ICD9_DGNS_CD_7_INP_CAT",
    "ICD9_DGNS_CD_8_INP_CAT",
    "ICD9_DGNS_CD_9_INP_CAT",
]

other_dropdown_fields = [
    "BENE_RACE_CD",
    "BENE_SEX_IDENT_CD",
    "BENE_ESRD_IND",
    "PRVDR_NUM_CAT",
    "ICD9_PRCDR_CD_1_INP_CAT",
]
#%%
diagnosid_code_category = (
    inpatient_target[icd_diagnosis_fields].melt()["value"].unique()
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
                            children=[f"ENTER {_}"],
                            id=f"{_}_LABEL",
                            className="input__heading",
                        ),
                        dcc.Input(
                            id=f"{_}",
                            type="number",
                            placeholder=f"{_}",
                            debounce=True,
                            className="oper__input",
                        ),
                    ],
                    className="input__container",
                )
                for _ in numerical_fields
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            children=[f"SELECT IF PATIENT HAS {_}"],
                            id=f"{_}_LABEL",
                            className="input__heading",
                        ),
                        dcc.Dropdown(
                            id=f"{_}",
                            options=map_comorbity,
                            className="reag__select",
                            placeholder=f"Select {_}.",
                        ),
                    ],
                    className="dropdown__container",
                )
                for _ in comorbidity_field
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
                                    children=[f"SELECT {_} FOR THE PATIENT"],
                                    id=f"{_}_LABEL",
                                    className="input__heading",
                                ),
                                dcc.Dropdown(
                                    id=f"{_}",
                                    options=map_diagnosis_code,
                                    className="reag__select",
                                    placeholder=f"Select {_} FOR THE PATIENT.",
                                ),
                            ],
                            className="dropdown__container",
                        )
                        for _ in icd_diagnosis_fields
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
