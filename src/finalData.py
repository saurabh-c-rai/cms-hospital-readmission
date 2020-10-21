# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


import json

#%%
from utils.ExtractData import FetchSubset

# %%
import sqlite3

# %%
from tempfile import mkdtemp

import numpy as np
import pandas as pd

# %%
import seaborn as sns
from joblib import Memory, dump, load
from matplotlib import pyplot as plt

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
EDA_REPORT_LOCATION = config[1]["eda_report_location"]
MODEL_REPOSITORY_LOCATION = config[1]["model_repository_location"]


# %%
# import os

# cd_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(cd_path)


# %%
from imblearn import pipeline as imb_pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.compose import ColumnTransformer

# %%
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer

# %%
from sklearn.linear_model import LogisticRegression

# %%
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from CustomPipeline import (
    CardinalityReducer,
    SelectColumnsTransfomer,
    get_ct_feature_names,
)
from feature_importance import FeatureImportance

# %%
from StatisticalTest import ChiSquare

# %%
conn_object = sqlite3.connect("../database/cms_data.db")


# %%
query_string = """SELECT inp.DESYNPUF_ID, inp.CLM_FROM_DT AS CLM_FROM_DT_INP, inp.CLM_THRU_DT AS CLM_THRU_DT_INP, inp.PRVDR_NUM AS PRVDR_NUM_INP, inp.CLM_PMT_AMT AS CLM_PMT_AMT_INP, inp.NCH_PRMRY_PYR_CLM_PD_AMT AS NCH_PRMRY_PYR_CLM_PD_AMT_INP, inp.AT_PHYSN_NPI AS AT_PHYSN_NPI_INP, inp.OP_PHYSN_NPI AS OP_PHYSN_NPI_INP, inp.CLM_ADMSN_DT AS CLM_ADMSN_DT_INP, inp.CLM_PASS_THRU_PER_DIEM_AMT AS CLM_PASS_THRU_PER_DIEM_AMT_INP, inp.NCH_BENE_IP_DDCTBL_AMT AS NCH_BENE_IP_DDCTBL_AMT_INP, inp.NCH_BENE_PTA_COINSRNC_LBLTY_AM AS NCH_BENE_PTA_COINSRNC_LBLTY_AM_INP, inp.NCH_BENE_BLOOD_DDCTBL_LBLTY_AM AS NCH_BENE_BLOOD_DDCTBL_LBLTY_AM_INP, inp.CLM_UTLZTN_DAY_CNT AS CLM_UTLZTN_DAY_CNT_INP, inp.NCH_BENE_DSCHRG_DT AS NCH_BENE_DSCHRG_DT_INP, inp.CLM_DRG_CD AS CLM_DRG_CD_INP, inp.PRVDR_NUM_CAT AS PRVDR_NUM_CAT_INP, inp.Next_CLM_ADMSN_DT AS Next_CLM_ADMSN_DT_INP, 
inp.Readmission_within_30days AS Readmission_within_30days_INP, inp.CLAIM_YEAR AS CLAIM_YEAR_INP, inp.ADMTNG_ICD9_DGNS_CD_CAT AS ADMTNG_ICD9_DGNS_CD_CAT_INP, inp.ICD9_DGNS_CD_1_CAT AS ICD9_DGNS_CD_1_CAT_INP, inp.ICD9_DGNS_CD_2_CAT AS ICD9_DGNS_CD_2_CAT_INP, inp.ICD9_DGNS_CD_3_CAT AS ICD9_DGNS_CD_3_CAT_INP, inp.ICD9_DGNS_CD_4_CAT AS ICD9_DGNS_CD_4_CAT_INP, 
inp.ICD9_DGNS_CD_5_CAT AS ICD9_DGNS_CD_5_CAT_INP, inp.ICD9_DGNS_CD_6_CAT AS ICD9_DGNS_CD_6_CAT_INP, inp.ICD9_DGNS_CD_7_CAT AS ICD9_DGNS_CD_7_CAT_INP, inp.ICD9_DGNS_CD_8_CAT AS ICD9_DGNS_CD_8_CAT_INP, inp.ICD9_DGNS_CD_9_CAT AS ICD9_DGNS_CD_9_CAT_INP, inp.ICD9_PRCDR_CD_1_CAT AS ICD9_PRCDR_CD_1_CAT_INP
FROM Inpatient_claims_2 inp
"""


# %%
claim_data = pd.read_sql_query(
    query_string,
    con=conn_object,
    parse_dates={
        "CLM_FROM_DT_INP": {"format": "%Y-%m-%d"},
        "CLM_THRU_DT_INP": {"format": "%Y-%m-%d"},
        "CLM_ADMSN_DT_INP": {"format": "%Y-%m-%d"},
        "NCH_BENE_DSCHRG_DT_INP": {"format": "%Y-%m-%d"},
        "Next_CLM_ADMSN_DT_INP": {"format": "%Y-%m-%d"},
        # "CLM_FROM_DT_OUT": {"format": "%Y-%m-%d"},
        # "CLM_THRU_DT_OUT": {"format": "%Y-%m-%d"},
    },
)


# %%
claim_data.head()


# %%
claim_data.columns


# %%
claim_data["Readmission_within_30days_INP"].value_counts() / claim_data.shape[0]


# %%
# claim_data = claim_data[(claim_data['NCH_BENE_DSCHRG_DT_INP'] <= claim_data['CLM_FROM_DT_OUT']) & (claim_data['CLM_FROM_DT_OUT'] <= claim_data['Next_CLM_ADMSN_DT_INP'])]


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
final_df["Readmission_within_30days_INP"].value_counts() / final_df.shape[0]


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
    # "PRVDR_NUM_CAT_OUT",
    # "HCPCS_CD_1_CAT_OUT",
    # "HCPCS_CD_2_CAT_OUT",
    # "HCPCS_CD_3_CAT_OUT",
    # "ICD9_DGNS_CD_1_CAT_OUT",
    # "ICD9_DGNS_CD_2_CAT_OUT",
    "Readmission_within_30days_INP",
    # "BENE_AGE_CAT"
    # "AT_PHYSN_NPI_OUT",
    # "AT_PHYSN_NPI_INP",
    # "OP_PHYSN_NPI_INP"
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
    # "PRVDR_NUM_OUT",
    # "ICD9_DGNS_CD_1_OUT",
    # "ICD9_DGNS_CD_2_OUT",
    # "HCPCS_CD_1_OUT",
    # "HCPCS_CD_2_OUT",
    # "HCPCS_CD_3_OUT",
    # "HCPCS_CD_1_CAT_DESC_OUT",
    # "HCPCS_CD_2_CAT_DESC_OUT",
    # "HCPCS_CD_3_CAT_DESC_OUT",
]
date_cols = list(final_df.select_dtypes(include="datetime").columns)
npi_cols = [col for col in final_df.select_dtypes(include="number") if "NPI" in col]


# %%
df = final_df.copy()


# %%
# df.to_sql("final_readmission_df", con=conn_object, index=False, if_exists="replace")


# %%
df[categorical_features] = df[categorical_features].astype("category")


# %%
df.drop(columns=cols_to_drop + date_cols + npi_cols, inplace=True, axis=1)


# %%
df.dtypes

#%%
# Check - 1 dropping columns as per feature importance
# df.drop(columns="BENRES_IP", inplace=True, axis=1)
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
#     correlations={
#         "pearson": {"calculate": True},
#         "spearman": {"calculate": True},
#         "kendall": {"calculate": True},
#         "phi_k": {"calculate": True},
#         "cramers": {"calculate": True},
#     },
# )
# eda_report.to_file(f"{EDA_REPORT_LOCATION}\\final_df_eda_report.html")


# %%
X = df.drop(columns=["Readmission_within_30days_INP"], axis=1)
y = df.loc[:, "Readmission_within_30days_INP"]


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE_FOR_SPLIT)


# %%
categorical_features = list(X.select_dtypes(include="category").columns)
numerical_features = list(X.select_dtypes(include="number").columns)


# %%
ct = ChiSquare(pd.concat([X_train, y_train], axis=1))


# %%
# cramers = pd.DataFrame(
#     {
#         i: [ct.cramers_v(i, j) for j in categorical_features]
#         for i in categorical_features
#     }
# )
# cramers["column"] = [i for i in categorical_features if i not in ["memberid"]]
# cramers.set_index("column", inplace=True)


# # %%
# # categorical correlation heatmap
# plt.figure(figsize=(25, 25))
# sns.heatmap(cramers, annot=True, fmt=".2f", cmap="magma")
# plt.show()

# # > High correlation between BENE_STATE_COUNTY_CODE & PRVDR_NUM_CAT_INP columns

# # %%
# # X_train.drop(columns=["BENE_STATE_COUNTY_CODE"], axis=1, inplace=True)
# # X_test.drop(columns=["BENE_STATE_COUNTY_CODE"], axis=1, inplace=True)


# # %%
# plt.figure(figsize=(25, 25))
# sns.heatmap(
#     X_train.select_dtypes(include="number").corr(), annot=True, fmt=".2f", cmap="magma"
# )
# plt.show()
# #%%
# distribution_age = pd.crosstab(
#     pd.cut(df["BENE_AGE"], bins=5, labels=["25-40", "40-55", "55-70", "70-85", "85+"]),
#     df["Readmission_within_30days_INP"],
#     normalize="index",
# )


# #%%
# plt.figure(figsize=(10, 7))
# sns.barplot(distribution_age.index, distribution_age[1])
# sns.barplot(distribution_age.index, distribution_age[0])
# plt.xticks(rotation=90)


# %%
# # Very high correlation between ('BENRES_OP', 'MEDREIMB_OP') & ('BENRES_CAR', 'MEDREIMB_CAR') hence dropping one column in the pair

X_train.drop(columns=["MEDREIMB_OP", "MEDREIMB_CAR"], inplace=True)
X_test.drop(columns=["MEDREIMB_OP", "MEDREIMB_CAR"], inplace=True)


# %%
X_train.columns


# %%

for c in categorical_features:
    ct.TestIndependence(c, "Readmission_within_30days_INP")


# %%
# X_train.drop(columns=ct.dfIrreleventCols, inplace=True)
# X_test.drop(columns=ct.dfIrreleventCols, inplace=True)


# %%
categorical_features = list(X_train.select_dtypes(include="category").columns)
numerical_feature = list(X_train.select_dtypes(include="number").columns)


# %%
X_train.columns
X_test.columns


# %%
# Preprocessing objects
std_scalar = StandardScaler()
min_max_scalar = MinMaxScaler()
onehot_encoder = OneHotEncoder(drop="first", sparse=False)
median_imputer = SimpleImputer(strategy="median", missing_values=np.nan)
constant_imputer = SimpleImputer(
    strategy="constant", fill_value=MISSING_VALUE_LABEL, missing_values=np.nan
)
ordinal_encoder = OrdinalEncoder()


# %%
numerical_cachedir = mkdtemp()
numerical_memory = Memory(location=numerical_cachedir, verbose=10)
catergorical_cachedir = mkdtemp()
categorical_memory = Memory(location=numerical_cachedir, verbose=10)


# %%
numerical_transformer = Pipeline(
    steps=[("imputer_with_medium", median_imputer), ("scaler", std_scalar)],
    verbose=True,
    memory=numerical_memory,
)


# %%
categorical_transformer = Pipeline(
    steps=[
        ("imputer_with_constant", constant_imputer),
        # (
        #     "infrequent_category_remover",
        #     CardinalityReducer(
        #         cutt_off=INFREQUENT_CATEGORY_CUT_OFF, label=INFREQUENT_CATEGORY_LABEL
        #     ),
        # ),
        ("onehot", onehot_encoder),
    ],
    verbose=True,
    memory=categorical_memory,
)


# %%
ord_categorical_transformer = Pipeline(
    steps=[
        ("imputer_with_constant", constant_imputer),
        (
            "infrequent_category_remover",
            CardinalityReducer(
                cutt_off=INFREQUENT_CATEGORY_CUT_OFF, label=INFREQUENT_CATEGORY_LABEL
            ),
        ),
        ("ordinal", ordinal_encoder),
    ],
    verbose=True,
    memory=categorical_memory,
)


# %%
preprocessing_pipeline = ColumnTransformer(
    [
        ("categorical", categorical_transformer, categorical_features),
        ("numerical", numerical_transformer, numerical_feature),
    ],
    remainder="drop",
    verbose=True,
)


# %%
X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
X_test_transformed = preprocessing_pipeline.transform(X_test)


# %%
X_train_transformed = pd.DataFrame(
    X_train_transformed, columns=get_ct_feature_names(preprocessing_pipeline)
)
X_test_transformed = pd.DataFrame(
    X_test_transformed, columns=get_ct_feature_names(preprocessing_pipeline)
)


# %%
X_train_transformed.head()


# %%
y_train = y_train.reset_index().drop(columns="index", axis=1).values.ravel()
y_test = y_test.reset_index().drop(columns="index", axis=1).values.ravel()

# %%
# Model initialization
lr = LogisticRegression(random_state=RANDOM_STATE)
lda = LinearDiscriminantAnalysis()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
gnb = GaussianNB()
rfc = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
#%%
# %%
# Oversampler initialization
smt = SMOTE(random_state=RANDOM_STATE, n_jobs=-1)
random_oversampling = RandomOverSampler(random_state=RANDOM_STATE)

#%%
# Feature selector initialization
rfe_lr = RFE(estimator=lr, n_features_to_select=20)
rfe_dt = RFE(estimator=dt, n_features_to_select=20)

#%%
logreg_rfe_pipeline = Pipeline(
    [
        # ("Select Relevent Columns", SelectColumnsTransfomer(columns=numerical_feature + ct.dfReleventCols)),
        ("Preprocessing Step", preprocessing_pipeline),
        ("Feature Selection", rfe_lr),
        ("LogReg_Classifier", lr),
    ]
)

logreg_rfe_smote_pipeline = Pipeline(
    [
        # ("Select Relevent Columns", SelectColumnsTransfomer(columns=numerical_feature + ct.dfReleventCols)),
        ("Preprocessing Step", preprocessing_pipeline),
        ("SMOTE oversampling", smt),
        ("Feature Selection", rfe_lr),
        ("LogReg_Classifier", lr),
    ]
)

# %%
num_folds = 5
seed = 7
scoring = "f1"
models = []
models.append(("LR", lr))
models.append(("LDA", lda))
models.append(("KNN", knn))
models.append(("CART", dt))
models.append(("NB", gnb))


# %%
dt_pipeline = Pipeline(
    [
        # ("Select Relevent Columns", SelectColumnsTransfomer(columns=numerical_feature + ct.dfReleventCols)),
        ("Preprocessing Step", preprocessing_pipeline),
        ("DT_Classifier", dt),
    ]
)

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
def getMetricsData(y_true, y_pred):
    print(f"Accuracy score is {accuracy_score(y_true, y_pred)}")
    print(f"Precision score is {precision_score(y_true, y_pred)}")
    print(f"Recall score is {recall_score(y_true, y_pred)}")
    print(f"F1 score is {f1_score(y_true, y_pred)}")
    print(f"AUC ROC score is {roc_auc_score(y_true, y_pred)}")
    print(f"Classification Report is \n {classification_report(y_true, y_pred)}")
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap="Blues", fmt="d")
    plt.show()


# %%
def fit_predict(pipeline):
    model = pipeline.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{"="*40} {pipeline.steps[-1][-1]} {"="*50}')
    print(f'{"="*40} {"Training Metrics"} {"="*50}')
    getMetricsData(y_train, model.predict(X_train))
    print(f'{"="*40} {"Testing Metrics"} {"="*50}')
    getMetricsData(y_test, y_pred)


# %%
fit_predict(dt_pipeline)


# %%
# dump(dt_pipeline, f'{MODEL_REPOSITORY_LOCATION}\decisionTreePipeline.pkl', compress=1)
# dump(dt_pipeline[-1], f'{MODEL_REPOSITORY_LOCATION}\decisionTreeClassifier.pkl', compress=1)


# %%
feature_importance = FeatureImportance(dt_pipeline)
feature_importance.plot(top_n_features=50, width=1000)


# %%
fit_predict(dt_smote_pipeline)


# %%
feature_importance = FeatureImportance(dt_smote_pipeline)
feature_importance.plot(top_n_features=50, width=1000)


# %%
fit_predict(dt_randover_pipeline)


# %%
feature_importance = FeatureImportance(dt_randover_pipeline)
feature_importance.plot(top_n_features=50, width=1000)


# %%
pca = PCA(n_components=0.95)
ipca = IncrementalPCA(n_components=0.95)

#%%
# # %%
# parameter_grid = {
#     "DT_Classifier__max_depth": np.linspace(1, 32, 32, endpoint=True),
#     "DT_Classifier__min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),
#     "DT_Classifier__min_samples_split": np.linspace(0.1, 1.0, 10, endpoint=True),
#     "DT_Classifier__max_features": ["auto", "log2", None],
# }


# # %%
# rand_grid_search = RandomizedSearchCV(
#     estimator=dt_pipeline,
#     param_distributions=parameter_grid,
#     n_iter=15,
#     scoring="f1",
#     verbose=4,
# )

# #%%
# rand_grid_search.fit(X_train, y_train)
# %%
num_folds = 5
seed = 7
scoring = "f1"
models = []
models.append(("LR", lr))
models.append(("LDA", lda))
models.append(("KNN", knn))
models.append(("CART", dt))
models.append(("NB", gnb))


# %%
# results = []
# names = []
# for name, model in models:
#     kfold = KFold(n_splits=num_folds, random_state=seed)
#     pipeline = Pipeline([("Preprocessing Step", preprocessing_pipeline), ("DT Classifier", model)])
#     cv_results = cross_val_score(
#         pipeline, X_train, y_train, cv=kfold, scoring=scoring
#     )
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s %f %f " % (name, cv_results.mean(), cv_results.std())
#     print(msg)

# # Ensembling Classifiers

# %%
extratrees_clf = ExtraTreesClassifier(
    max_features=0.7363742386320187,
    n_estimators=80,
    n_jobs=-1,
    random_state=4,
    verbose=False,
)


# %%
extratrees_clf_pipeline = Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("Extratrees_Classifier", extratrees_clf),
    ]
)

extratrees_smt_pipeline = imb_pipeline.Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("SMOTE oversampling", smt),
        ("Extratrees_Classifier", extratrees_clf),
    ]
)


extratrees_randover_pipeline = imb_pipeline.Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("Random oversampling", random_oversampling),
        ("Extratrees_Classifier", extratrees_clf),
    ]
)

# %%
fit_predict(extratrees_clf_pipeline)

#%%
fit_predict(extratrees_smt_pipeline)

#%%
fit_predict(extratrees_randover_pipeline)


# %%


# %%
rfc_model_1 = RandomForestClassifier(
    bootstrap=False,
    max_features=0.40810278517436016,
    n_estimators=73,
    n_jobs=-1,
    random_state=1,
    verbose=False,
)


# %%
rfc_clf_pipeline = Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("Extratrees_Classifier", rfc_model_1),
    ]
)

rfc_smt_pipeline = imb_pipeline.Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("SMOTE oversampling", smt),
        ("Extratrees_Classifier", rfc_model_1),
    ]
)

rfc_randover_pipeline = imb_pipeline.Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("random oversampling", random_oversampling),
        ("Extratrees_Classifier", rfc_model_1),
    ]
)


# %%
fit_predict(rfc_clf_pipeline)

#%%
fit_predict(rfc_smt_pipeline)

#%%
fit_predict(rfc_randover_pipeline)


# %%
# dump(rfc_model_1, f'{MODEL_REPOSITORY_LOCATION}\RandomForestEstimator.joblib')

# dump(rfc_clf_pipeline, f'{MODEL_REPOSITORY_LOCATION}\RandomForestPipeline.joblib', compress=1)

# %%
estimators = [
    ("dt_pipeline", dt_pipeline),
    ("rfc_pipeline", rfc_clf_pipeline),
    ("extratrees_pipeline", extratrees_clf_pipeline),
]

estimators_smote = [
    ("rfc_smote_pipeline", rfc_smt_pipeline),
    ("extratrees_smote_pipeline", extratrees_smt_pipeline),
]
# %%
voting_clf = VotingClassifier(
    estimators=estimators, voting="soft", verbose=True, n_jobs=-1
)


# %%
voting_clf.fit(X_train, y_train)


# %%
y_pred = voting_clf.predict(X_test)


# %%
getMetricsData(y_true=y_test, y_pred=y_pred)


# %%
# dump(voting_clf, f'{MODEL_REPOSITORY_LOCATION}\VotingClassifier.pkl', compress=1)


# %%
bagg_clf = BaggingClassifier(
    base_estimator=rfc_clf_pipeline.steps[-1][-1],
    n_estimators=10,
    random_state=RANDOM_STATE,
    max_samples=0.8,
    max_features=0.8,
    bootstrap_features=False,
    bootstrap=True,
)


# %%
bagg_clf.fit(X_train_transformed, y_train)


# %%
y_pred = bagg_clf.predict(X_test_transformed)


# %%
getMetricsData(y_test, y_pred)


# %%
# rf_defaults = {'n_jobs': -1,
#                'random_state': 100}

# space4nested = hp.pchoice('classifier_type',
#                           [(0.4, {
#                                   'min_samples_leaf': pyll.scope.int(pyll.scope.maximum(hp.qlognormal('min_samples_leaf', 2, 1.2, 1), 1)),
#                                   'max_features': hp.choice('max_features', [1.0, 0.5, 'sqrt', 'log2']),
#                                   'n_estimators':hp.choice('n_estimators',np.arange(100, 500+1,50, dtype=int)),
#                                   'max_depth':hp.uniform('max_depth',5,20),
#                                   'min_samples_split':hp.choice('min_samples_split',np.arange(2, 6+1,1, dtype=int))

#                                   }
#                             )]
#                           )

# def hyperopt_rfc_nested(params):
#     params = {**rf_defaults, **params}
#     rf_model = RandomForestClassifier(**params)
#     rf_model.fit(X_train_transformed, y_train)
#     return_dict = {'loss': -roc_auc_score(y_test, rf_model.predict_proba(X_test_transformed)[:, 1]),
#                    'status': STATUS_OK}
#     return return_dict


# %%
# trials = Trials()

# best = fmin(hyperopt_rfc_nested, space4nested, algo=tpe.suggest, max_evals=100, trials=trials)

# print(f'best: {space_eval(space4nested, best)}')


# %%
conn_object.close()

# %%
from hpsklearn import HyperoptEstimator, extra_trees, random_forest
from hyperopt import tpe

# %%
estim_extratrees = HyperoptEstimator(
    classifier=extra_trees("extratrees_clf"),
    preprocessing=[],
    algo=tpe.suggest,
    max_evals=10,
    trial_timeout=600,
)

estim_extratrees.fit(X_train_transformed.to_numpy(), y_train)

estim_extratrees.score(X_test_transformed.to_numpy(), y_test)

estim_extratrees.best_model()["learner"]

#%%
extratrees_hp_model = estim_extratrees.best_model()["learner"]
# %%
estim_rfc = HyperoptEstimator(
    classifier=random_forest("rfc_clf"),
    preprocessing=[],
    algo=tpe.suggest,
    max_evals=10,
    trial_timeout=600,
)

estim_rfc.fit(X_train_transformed.to_numpy(), y_train)

estim_rfc.score(X_test_transformed.to_numpy(), y_test)

estim_rfc.best_model()["learner"]

rfc_hp_model = estim_rfc.best_model()["learner"]
