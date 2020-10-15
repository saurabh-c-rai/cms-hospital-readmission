# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


# %%
import seaborn as sns
import numpy as np
import pandas as pd


# %%
import sqlite3
import json


# %%
with open("../config.json", "r") as f:
    config = json.load(f)


# %%
for dictionary in config:
    dictionary.items()


# %%
# constant variables
TEST_SIZE_FOR_SPLIT = config[3]["TEST_SIZE_FOR_SPLIT"]
INFREQUENT_CATEGORY_CUT_OFF = config[3]["INFREQUENT_CATEGORY_CUT_OFF"]
INFREQUENT_CATEGORY_LABEL = config[3]["INFREQUENT_CATEGORY_LABEL"]
RANDOM_STATE = config[3]["RANDOM_STATE"]
MISSING_VALUE_LABEL = config[3]["MISSING_VALUE_LABEL"]


# %%
# import os

# cd_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(cd_path)


# %%
from StatisticalTest import ChiSquare
from CustomPipeline import CardinalityReducer, get_ct_feature_names


# %%
from imblearn import pipeline as imb_pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler


# %%
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score


# %%
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


# %%
conn_object = sqlite3.connect("../database/cms_data.db")


# %%
query_string = """SELECT inp.DESYNPUF_ID, inp.CLM_FROM_DT AS CLM_FROM_DT_INP, inp.CLM_THRU_DT AS CLM_THRU_DT_INP, inp.PRVDR_NUM AS PRVDR_NUM_INP, inp.CLM_PMT_AMT AS CLM_PMT_AMT_INP, inp.NCH_PRMRY_PYR_CLM_PD_AMT AS NCH_PRMRY_PYR_CLM_PD_AMT_INP, inp.AT_PHYSN_NPI AS AT_PHYSN_NPI_INP, inp.OP_PHYSN_NPI AS OP_PHYSN_NPI_INP, inp.CLM_ADMSN_DT AS CLM_ADMSN_DT_INP, inp.CLM_PASS_THRU_PER_DIEM_AMT AS CLM_PASS_THRU_PER_DIEM_AMT_INP, inp.NCH_BENE_IP_DDCTBL_AMT AS NCH_BENE_IP_DDCTBL_AMT_INP, inp.NCH_BENE_PTA_COINSRNC_LBLTY_AM AS NCH_BENE_PTA_COINSRNC_LBLTY_AM_INP, inp.NCH_BENE_BLOOD_DDCTBL_LBLTY_AM AS NCH_BENE_BLOOD_DDCTBL_LBLTY_AM_INP, inp.CLM_UTLZTN_DAY_CNT AS CLM_UTLZTN_DAY_CNT_INP, inp.NCH_BENE_DSCHRG_DT AS NCH_BENE_DSCHRG_DT_INP, inp.CLM_DRG_CD AS CLM_DRG_CD_INP, inp.PRVDR_NUM_CAT AS PRVDR_NUM_CAT_INP, inp.Next_CLM_ADMSN_DT AS Next_CLM_ADMSN_DT_INP, 
inp.Readmission_within_30days AS Readmission_within_30days_INP, inp.CLAIM_YEAR AS CLAIM_YEAR_INP, inp.ADMTNG_ICD9_DGNS_CD_CAT AS ADMTNG_ICD9_DGNS_CD_CAT_INP, inp.ICD9_DGNS_CD_1_CAT AS ICD9_DGNS_CD_1_CAT_INP, inp.ICD9_DGNS_CD_2_CAT AS ICD9_DGNS_CD_2_CAT_INP, inp.ICD9_DGNS_CD_3_CAT AS ICD9_DGNS_CD_3_CAT_INP, inp.ICD9_DGNS_CD_4_CAT AS ICD9_DGNS_CD_4_CAT_INP, 
inp.ICD9_DGNS_CD_5_CAT AS ICD9_DGNS_CD_5_CAT_INP, inp.ICD9_DGNS_CD_6_CAT AS ICD9_DGNS_CD_6_CAT_INP, inp.ICD9_DGNS_CD_7_CAT AS ICD9_DGNS_CD_7_CAT_INP, inp.ICD9_DGNS_CD_8_CAT AS ICD9_DGNS_CD_8_CAT_INP, inp.ICD9_DGNS_CD_9_CAT AS ICD9_DGNS_CD_9_CAT_INP, inp.ICD9_PRCDR_CD_1_CAT AS ICD9_PRCDR_CD_1_CAT_INP, 
out.CLM_FROM_DT as CLM_FROM_DT_OUT, out.CLM_THRU_DT as CLM_THRU_DT_OUT, out.PRVDR_NUM as PRVDR_NUM_OUT, out.CLM_PMT_AMT as CLM_PMT_AMT_OUT, 
out.NCH_PRMRY_PYR_CLM_PD_AMT as NCH_PRMRY_PYR_CLM_PD_AMT_OUT, out.AT_PHYSN_NPI as AT_PHYSN_NPI_OUT, out.NCH_BENE_BLOOD_DDCTBL_LBLTY_AM as NCH_BENE_BLOOD_DDCTBL_LBLTY_AM_OUT, 
out.ICD9_DGNS_CD_1 as ICD9_DGNS_CD_1_OUT, out.ICD9_DGNS_CD_2 as ICD9_DGNS_CD_2_OUT, out.NCH_BENE_PTB_DDCTBL_AMT as NCH_BENE_PTB_DDCTBL_AMT_OUT, 
out.NCH_BENE_PTB_COINSRNC_AMT as NCH_BENE_PTB_COINSRNC_AMT_OUT, out.HCPCS_CD_1 as HCPCS_CD_1_OUT, out.HCPCS_CD_2 as HCPCS_CD_2_OUT, out.HCPCS_CD_3 as HCPCS_CD_3_OUT, out.PRVDR_NUM_CAT as PRVDR_NUM_CAT_OUT, OUT.ICD9_DGNS_CD_1_CAT AS ICD9_DGNS_CD_1_CAT_OUT,OUT.ICD9_DGNS_CD_2_CAT AS ICD9_DGNS_CD_2_CAT_OUT,OUT.HCPCS_CD_1_CAT AS HCPCS_CD_1_CAT_OUT, OUT.HCPCS_CD_2_CAT AS HCPCS_CD_2_CAT_OUT,
OUT.HCPCS_CD_3_CAT AS HCPCS_CD_3_CAT_OUT, OUT.HCPCS_CD_1_CAT_DESC AS HCPCS_CD_1_CAT_DESC_OUT, OUT.HCPCS_CD_2_CAT_DESC AS HCPCS_CD_2_CAT_DESC_OUT, OUT.HCPCS_CD_3_CAT_DESC AS HCPCS_CD_3_CAT_DESC_OUT
FROM Inpatient_claims_2 inp LEFT JOIN Outpatient_claims_2 out 
on inp.DESYNPUF_ID = out.DESYNPUF_ID 
WHERE inp.NCH_BENE_DSCHRG_DT <= out.CLM_FROM_DT AND inp.Next_CLM_ADMSN_DT >= out.CLM_FROM_DT"""


# %%
claim_data = pd.read_sql_query(
    query_string,
    con=conn_object,
    parse_dates={
        "CLM_FROM_DT_INP": {"format": "%Y-%m-%d"},
        "CLM_THRU_DT_INP": {"format": "%Y-%m-%d"},
        "CLM_ADMSN_DT_INP": {"format": "%Y-%m-%d"},
        "NCH_BENE_DSCHRG_DT_INP": {"format": "%Y-%m-%d"},
        "Next_CLM_ADMSN_DT_IN": {"format": "%Y-%m-%d"},
        "CLM_FROM_DT_OUT": {"format": "%Y-%m-%d"},
        "CLM_THRU_DT_OUT": {"format": "%Y-%m-%d"},
    },
)


# %%
claim_data["Next_CLM_ADMSN_DT_INP"] = pd.to_datetime(
    claim_data["Next_CLM_ADMSN_DT_INP"], infer_datetime_format=True
)


# %%
claim_data.head()


# %%
claim_data.columns


# %%
claim_data.dtypes


# %%
claim_data.shape


# %%
beneficiary_summary_2 = pd.read_sql_query(
    "select * from Beneficiary_Data_2",
    con=conn_object,
    parse_dates=["BENE_BIRTH_DT", "BENE_DEATH_DT"],
)
beneficiary_summary_2.drop(columns=["index"], inplace=True)


# %%
beneficiary_summary_2.shape


# %%
beneficiary_summary_2.dtypes


# %%
final_df = pd.merge(
    left=beneficiary_summary_2,
    right=claim_data,
    left_on=["DESYNPUF_ID", "Year"],
    right_on=["DESYNPUF_ID", "CLAIM_YEAR_INP"],
    how="inner",
)


# %%
final_df.head()


# %%
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
    "PRVDR_NUM_CAT_OUT",
    "HCPCS_CD_1_CAT_OUT",
    "HCPCS_CD_2_CAT_OUT",
    "HCPCS_CD_3_CAT_OUT",
    "ICD9_DGNS_CD_1_CAT_OUT",
    "ICD9_DGNS_CD_2_CAT_OUT",
    "Readmission_within_30days_INP",
]


# %%
# columns to drop:
# PRVDR_NUM_INP : Category column exist PRVDR_NUM_CAT_INP
# CLM_DRG_CD_INP : Claim Diagnosis Related Group Code not relevant for Readmission detection
# PRVDR_NUM_OUT : Category column exists PRVDR_NUM_CAT_OUT
# 'ICD9_DGNS_CD_1_OUT', 'ICD9_DGNS_CD_2_OUT', 'HCPCS_CD_1_OUT', 'HCPCS_CD_2_OUT', 'HCPCS_CD_3_OUT' : Category column exists
# 'HCPCS_CD_1_CAT_DESC_OUT', 'HCPCS_CD_2_CAT_DESC_OUT', 'HCPCS_CD_3_CAT_DESC_OUT' : Description column to be used later

cols_to_drop = [
    "Year",
    "CLAIM_YEAR_INP",
    "PRVDR_NUM_INP",
    "CLM_DRG_CD_INP",
    "PRVDR_NUM_OUT",
    "ICD9_DGNS_CD_1_OUT",
    "ICD9_DGNS_CD_2_OUT",
    "HCPCS_CD_1_OUT",
    "HCPCS_CD_2_OUT",
    "HCPCS_CD_3_OUT",
    "HCPCS_CD_1_CAT_DESC_OUT",
    "HCPCS_CD_2_CAT_DESC_OUT",
    "HCPCS_CD_3_CAT_DESC_OUT",
]
date_cols = list(final_df.select_dtypes(include="datetime").columns)


# %%
df = final_df.copy()


# %%
df[categorical_features] = df[categorical_features].astype("category")


# %%
df.drop(columns=cols_to_drop + date_cols, inplace=True, axis=1)


# %%
df.dtypes


# %%
# from pandas_profiling import ProfileReport

# eda_report = ProfileReport(
#     df,
#     title="Exploratory Data Analysis",
#     minimal=True,
#     interactions={"continuous": False},
#     missing_diagrams={
#         "bar": True,
#         "matrix": True,
#         "heatmap": True,
#         "dendrogram": True,
#     },
#     correlations= {
#             "pearson": {"calculate": True},
#             "spearman": {"calculate": True},
#             "kendall": {"calculate": True},
#             "phi_k": {"calculate": True},
#             "cramers": {"calculate": True},
#         },
# )
# eda_report.to_file("output.html")


# %%
ct = ChiSquare(df)


# %%
cramers = pd.DataFrame(
    {
        i: [ct.cramers_v(i, j) for j in categorical_features]
        for i in categorical_features
    }
)
cramers["column"] = [i for i in categorical_features if i not in ["memberid"]]
cramers.set_index("column", inplace=True)


# %%
# categorical correlation heatmap
from matplotlib import pyplot as plt

plt.figure(figsize=(25, 25))
sns.heatmap(cramers, annot=True, fmt=".2f", cmap="magma")
plt.show()


# %%

plt.figure(figsize=(25, 25))
sns.heatmap(
    df.select_dtypes(include="number").corr(), annot=True, fmt=".2f", cmap="magma"
)
# plt.show()
# #%%
# distribution_age = pd.crosstab(
#     pd.cut(df["BENE_AGE"], bins=5, labels=["25-40", "40-55", "55-70", "70-85", "85+"]),
#     df["Readmission_within_30days_INP"],
#     normalize="index",
# )
# #%%
# # Very high correlation between ('BENRES_OP', 'MEDREIMB_OP') & ('BENRES_CAR', 'MEDREIMB_CAR') hence dropping one column in the pair

df.drop(columns=["MEDREIMB_OP", "MEDREIMB_CAR"], inplace=True)


# #%%
# plt.figure(figsize=(10, 7))
# sns.barplot(distribution_age.index, distribution_age[1])
# sns.barplot(distribution_age.index, distribution_age[0])
# plt.xticks(rotation=90)


# %%
X = df.drop(columns=["Readmission_within_30days_INP"], axis=1)
y = df.loc[:, "Readmission_within_30days_INP"]


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE_FOR_SPLIT)


# %%
categorical_features = list(X.select_dtypes(include="category").columns)
numerical_feature = list(X.select_dtypes(include="number").columns)


# %%

for c in numerical_feature + categorical_features:
    ct.TestIndependence(c, "Readmission_within_30days_INP")


# %%
# Imputation objects
std_scalar = StandardScaler()
min_max_scalar = MinMaxScaler()
onehot_encoder = OneHotEncoder(drop="first", sparse=False)
median_imputer = SimpleImputer(strategy="median", missing_values=np.nan)
constant_imputer = SimpleImputer(
    strategy="constant", fill_value=MISSING_VALUE_LABEL, missing_values=np.nan
)


# %%
numerical_transformer = Pipeline(
    steps=[("imputer_with_medium", median_imputer), ("scaler", std_scalar)],
    verbose=True,
)


# %%
categorical_transformer = Pipeline(
    steps=[
        ("imputer_with_constant", constant_imputer),
        (
            "infrequent_category_remover",
            CardinalityReducer(
                cutt_off=INFREQUENT_CATEGORY_CUT_OFF, label=INFREQUENT_CATEGORY_LABEL
            ),
        ),
        ("onehot", onehot_encoder),
    ],
    verbose=True,
)


# %%
preprocessing_pipeline = ColumnTransformer(
    [
        ("categorical", categorical_transformer, categorical_features),
        ("numerical", numerical_transformer, numerical_feature),
    ],
    remainder="passthrough",
    verbose=True,
)


# %%
X_train_transformed = preprocessing_pipeline.fit_transform(X_train.iloc[:, 1:])
X_test_transformed = preprocessing_pipeline.transform(X_test.iloc[:, 1:])


# %%
X_train_transformed = pd.DataFrame(
    X_train_transformed, columns=get_ct_feature_names(preprocessing_pipeline)
)
X_test_transformed = pd.DataFrame(
    X_test_transformed, columns=get_ct_feature_names(preprocessing_pipeline)
)


# %%
num_folds = 5
seed = 7
scoring = "f1"
models = []
models.append(("LR", LogisticRegression()))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))


# %%
smt = SMOTE(random_state=RANDOM_STATE, n_jobs=-1)
random_oversampling = RandomOverSampler(random_state=RANDOM_STATE)


# %%
dt_smote_pipeline = imb_pipeline.Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("SMOTE Sampling", smt),
        ("Classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]
)

dt_randover_pipeline = imb_pipeline.Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("Random OverSampling", random_oversampling),
        ("Classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]
)


# %%
dt_smote_pipeline.fit(X_train.iloc[:, 1:], y_train)
dt_randover_pipeline.fit(X_train.iloc[:, 1:], y_train)


# %%
y_pred_smote = dt_smote_pipeline.predict(X_test.iloc[:, 1:])
y_pred_randover = dt_randover_pipeline.predict(X_test.iloc[:, 1:])


# %%
print(f"Accuracy score is {accuracy_score(y_test, y_pred_smote)}")
print(f"Precision score is {precision_score(y_test, y_pred_smote)}")
print(f"Recall score is {recall_score(y_test, y_pred_smote)}")
print(f"F1 score is {f1_score(y_test, y_pred_smote)}")
print(f"Classification Report is \n {classification_report(y_test, y_pred_smote)}")


# %%
sns.heatmap(
    confusion_matrix(y_true=y_test, y_pred=y_pred_smote),
    annot=True,
    cmap="magma",
    fmt="d",
)


# %%
print(f"Accuracy score is {accuracy_score(y_test, y_pred_randover)}")
print(f"Precision score is {precision_score(y_test, y_pred_randover)}")
print(f"Recall score is {recall_score(y_test, y_pred_randover)}")
print(f"F1 score is {f1_score(y_test, y_pred_randover)}")
print(f"Classification Report is \n {classification_report(y_test, y_pred_randover)}")
sns.heatmap(
    confusion_matrix(y_true=y_test, y_pred=y_pred_randover),
    annot=True,
    cmap="magma",
    fmt="d",
)


# %%
# results = []
# names = []
# for name, model in models:
#     kfold = KFold(n_splits=num_folds, random_state=seed)
#     cv_results = cross_val_score(
#         model, X_train_transformed, y_train, cv=kfold, scoring=scoring
#     )
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s %f %f " % (name, cv_results.mean(), cv_results.std())
#     print(msg)


# %%

