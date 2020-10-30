#%%
# To print output of all the lines
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

#%%
import json
import os
from tempfile import mkdtemp

#%%
# Computation packages
import numpy as np
import pandas as pd

#%%
# Imbalanced class handling packages
from imblearn import pipeline as imb_pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler

#%%
# Pipelines
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#%%
from utils import ExtractData

# %%
# Custom packages
from utils import (
    CategorizeCardinalData,
    StatisticalTest,
    feature_importance,
)
from utils.CustomPipeline import (
    CardinalityReducer,
    get_ct_feature_names,
    SelectColumnsTransfomer,
)

#%%
# Plotting packages
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px

#%%
from joblib import Memory, dump, load

#%%
# Pipelines & transformers
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

#%%
# module for dimensionality reduction
from sklearn.decomposition import PCA, IncrementalPCA
from prince import FAMD

#%%
# Preprocessing modules
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

#%%
# sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier

# %%
# Model metric methods
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)

#%%
# clustering
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering

#%%
# loading configuration values
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

# %%
data = ExtractData.FetchSubset(subset_list=[2])

# %%
data_inpatient_claims = data.fetchFromInpatientDataset()
# %%
data_inpatient_claims.isna().sum() / data_inpatient_claims.shape[0]
# %%
data_inpatient_claims.dropna(
    thresh=NON_NAN_THRESHOLD * data_inpatient_claims.shape[0], axis=1, inplace=True
)
# %%
data_inpatient_claims.columns
data_inpatient_claims.shape
data_inpatient_claims.isna().sum()

# %%
# Processing of PRVDR_NUM column. Converting the data into categories
data_inpatient_claims["PRVDR_NUM"].value_counts()
prvdr_num = CategorizeCardinalData.ProviderNumCategoryCreator()
prvdr_num.get_categories_for_providers(data_inpatient_claims["PRVDR_NUM"])
data_inpatient_claims = data_inpatient_claims.merge(
    right=prvdr_num.unique_prvdr_num_category_df, on=["PRVDR_NUM"], how="left"
)


# %%
diagnosis_code = [col for col in data_inpatient_claims.columns if "ICD9_DGNS" in col]
icd_procedural_code = [
    col for col in data_inpatient_claims.columns if "ICD9_PRCDR" in col
]
icd_hcpcs_code = [col for col in data_inpatient_claims.columns if "HCPCS_CD" in col]

#%%
# Preparing dataframe for categorizing
data_inpatient_claims[diagnosis_code + icd_procedural_code + icd_hcpcs_code] = (
    data_inpatient_claims[diagnosis_code + icd_procedural_code + icd_hcpcs_code]
    .astype("str")
    .replace(["nan", "na"], np.nan)
)

#%%
# Processing of Procdural Code columns
proc_code = CategorizeCardinalData.ProcedureCodeCategoryCreator()
proc_code.get_categories_for_procedure_code(data_inpatient_claims[icd_procedural_code])
for col in icd_procedural_code:
    data_inpatient_claims[f"{col}"] = data_inpatient_claims[f"{col}"].str[:2]
    data_inpatient_claims[f"{col}_CAT"] = pd.merge(
        left=data_inpatient_claims,
        right=proc_code.unique_procedure_code_category_df,
        left_on=col,
        right_on="Procedure_code",
        how="left",
    )["Procedure_code_CAT"]

# %%
# Processing of Diagnosis Code
diag_code = CategorizeCardinalData.DiagnosisCodeCategoryCreator()
diag_code.get_categories_for_diagnosis_code(data_inpatient_claims[diagnosis_code])
for col in diagnosis_code:
    data_inpatient_claims[f"{col}"] = data_inpatient_claims[f"{col}"].str[:3]
    data_inpatient_claims[f"{col}_CAT"] = pd.merge(
        left=data_inpatient_claims,
        right=diag_code.unique_diagnosis_code_category_df,
        left_on=col,
        right_on="Diagnosis_code",
        how="left",
    )["Description"]

# %%
# Create a year column for merging with beneficiary summary data
data_inpatient_claims["CLAIM_YEAR"] = data_inpatient_claims["CLM_ADMSN_DT"].dt.year

#%%
# selecting desease as 390-459 or Desease of Circulatory system since it has maximum data
diagnosis_code_cat = [
    col for col in data_inpatient_claims if ("_CAT" in col) & ("DGNS" in col)
]

selected_icd_code = "390-459"
selected_data = data_inpatient_claims.loc[
    data_inpatient_claims[diagnosis_code_cat].isin([selected_icd_code]).any(axis=1), :,
].copy()

#%%
# Generating target variable
selected_data.loc[:, "Next_CLM_ADMSN_DT"] = selected_data.groupby("DESYNPUF_ID")[
    "CLM_ADMSN_DT"
].shift(-1)

# Calculatings days between patient admission and the previous time the patient was discharged
selected_data.loc[:, "Readmission_Day_Count"] = (
    selected_data["Next_CLM_ADMSN_DT"] - selected_data["NCH_BENE_DSCHRG_DT"]
).apply(lambda x: x.days)

# if the Readmission_Day_Count <= 30, the Unplanned readmission is 1 or else 0
selected_data.loc[:, "IsReadmitted"] = (
    selected_data.loc[:, "Readmission_Day_Count"] <= 30
) * 1

selected_data.drop(columns="Readmission_Day_Count", inplace=True)
selected_data.sort_values(
    by=["DESYNPUF_ID", "NCH_BENE_DSCHRG_DT"], ascending=True, inplace=True
)
#%%
# Removing the not readmitted record for the patient which have atleast one admitted record
readmitted_patients_data = selected_data.loc[(selected_data["IsReadmitted"] == 1), :]
readmitted_patients = readmitted_patients_data.loc[:, "DESYNPUF_ID"].unique()

drop_index = selected_data.loc[
    (selected_data["DESYNPUF_ID"].isin(readmitted_patients))
    & (selected_data["IsReadmitted"] == 0)
].index

selected_data.drop(index=drop_index, axis=0, inplace=True)
readmitted_patients_data = selected_data.loc[(selected_data["IsReadmitted"] == 1), :]
not_readmitted_patients_data = selected_data.loc[
    (selected_data["IsReadmitted"] == 0), :
]

# Uncomment the below line to remove duplicate patient from non readmitted data
# not_readmitted_patients_data.drop_duplicates(subset=["DESYNPUF_ID", "IsReadmitted"], keep='last', inplace=True)

final_inpatient_data = pd.concat(
    [readmitted_patients_data, not_readmitted_patients_data], axis=0
)

#%%
final_inpatient_data.drop(columns=icd_procedural_code + diagnosis_code, inplace=True)

#%%
data_outpatient_claim = data.fetchFromOutpatientDataset()

#%%
data_outpatient_claim.dropna(
    thresh=NON_NAN_THRESHOLD * data_outpatient_claim.shape[0], axis=1, inplace=True
)
#%%
# data_outpatient_claim["PRVDR_NUM"].value_counts()
# prvdr_num.get_categories_for_providers(data_outpatient_claim["PRVDR_NUM"])
# data_outpatient_claim = data_outpatient_claim.merge(
#     right=prvdr_num.unique_prvdr_num_category_df, on=["PRVDR_NUM"], how="left"
# )

#%%
data_outpatient_claim.head()
#%%
diagnosis_code_out = [
    col for col in data_outpatient_claim.columns if "ICD9_DGNS" in col
]
icd_procedural_code_out = [
    col for col in data_outpatient_claim.columns if "ICD9_PRCDR" in col
]
icd_hcpcs_code_out = [col for col in data_outpatient_claim.columns if "HCPCS_CD" in col]

#%%
data_outpatient_claim[
    diagnosis_code_out + icd_procedural_code_out + icd_hcpcs_code_out
] = (
    data_outpatient_claim[
        diagnosis_code_out + icd_procedural_code_out + icd_hcpcs_code_out
    ]
    .astype("str")
    .replace(["nan", "na"], np.nan)
)

#%%
# Processing of Procdural Code columns
# proc_code.get_categories_for_procedure_code(
#     data_outpatient_claim[icd_procedural_code_out]
# )
# for col in icd_procedural_code_out:
#     data_outpatient_claim[f"{col}"] = data_outpatient_claim[f"{col}"].str[:2]
#     data_outpatient_claim[f"{col}_CAT"] = pd.merge(
#         left=data_outpatient_claim,
#         right=proc_code.unique_procedure_code_category_df,
#         left_on=col,
#         right_on="Procedure_code",
#         how="left",
#     )["Procedure_code_CAT"]

# %%
# Processing of Diagnosis Code
diag_code.get_categories_for_diagnosis_code(data_outpatient_claim[diagnosis_code_out])
for col in diagnosis_code_out:
    data_outpatient_claim[f"{col}"] = data_outpatient_claim[f"{col}"].str[:3]
    data_outpatient_claim[f"{col}_CAT"] = pd.merge(
        left=data_outpatient_claim,
        right=diag_code.unique_diagnosis_code_category_df,
        left_on=col,
        right_on="Diagnosis_code",
        how="left",
    )["Description"]
#%%
data_outpatient_claim.columns = [
    "DESYNPUF_ID" if col == "DESYNPUF_ID" else col + "_OUT"
    for col in data_outpatient_claim
]
#%%
final_claim_data = pd.merge(
    left=final_inpatient_data, right=data_outpatient_claim, on="DESYNPUF_ID", how="left"
)


#%%
span_data = final_claim_data[
    (final_claim_data["NCH_BENE_DSCHRG_DT"] <= final_claim_data["CLM_FROM_DT_OUT"])
    & (final_claim_data["CLM_FROM_DT_OUT"] <= final_claim_data["Next_CLM_ADMSN_DT"])
]
#%%
span_data["ICD9_DGNS_CD_1_MAP_OUT"] = span_data["ICD9_DGNS_CD_1_CAT_OUT"].apply(
    lambda x: "390-459" if x == "390-459" else "Others"
)

span_data["ICD9_DGNS_CD_2_MAP_OUT"] = span_data["ICD9_DGNS_CD_2_CAT_OUT"].apply(
    lambda x: "390-459" if x == "390-459" else "Others"
)
#%%
claim_data = span_data.groupby(["DESYNPUF_ID", "CLM_ID"]).count()
claim_data.reset_index(inplace=True)
claim_data.head()
claim_data = claim_data.loc[:, ["DESYNPUF_ID", "CLM_ID", "SEGMENT"]]
#%%
final_inpatient_data = pd.merge(
    left=final_inpatient_data,
    right=claim_data,
    on=["DESYNPUF_ID", "CLM_ID"],
    how="left",
)

final_inpatient_data["SEGMENT_y"].fillna(value=0, inplace=True)
# %%
data_beneficiary_2008 = data.fetchFromBeneficiaryDataset(year=2008)
data_beneficiary_2008["YEAR"] = pd.to_datetime("2008-12-31", infer_datetime_format=True)
data_beneficiary_2009 = data.fetchFromBeneficiaryDataset(year=2009)
data_beneficiary_2009["YEAR"] = pd.to_datetime("2009-12-31", infer_datetime_format=True)
data_beneficiary_2010 = data.fetchFromBeneficiaryDataset(year=2010)
data_beneficiary_2010["YEAR"] = pd.to_datetime("2010-12-31", infer_datetime_format=True)
#%%

# %%
combined_beneficiary_data_2 = pd.concat(
    [data_beneficiary_2008, data_beneficiary_2009, data_beneficiary_2010,], axis=0,
)


# %%
combined_beneficiary_data_2["Temp_Death_DT"] = np.where(
    combined_beneficiary_data_2["BENE_DEATH_DT"].isna(),
    combined_beneficiary_data_2["YEAR"],
    combined_beneficiary_data_2["BENE_DEATH_DT"],
)

combined_beneficiary_data_2["Temp_Death_DT"] = pd.to_datetime(
    combined_beneficiary_data_2["Temp_Death_DT"], infer_datetime_format=True
)

# Finding age from Death & Birth Date
combined_beneficiary_data_2["BENE_AGE"] = (
    combined_beneficiary_data_2["Temp_Death_DT"]
    - combined_beneficiary_data_2["BENE_BIRTH_DT"]
).astype("timedelta64[Y]")
combined_beneficiary_data_2.drop(columns=["Temp_Death_DT"], inplace=True)

#%%
combined_beneficiary_data_2["YEAR"] = combined_beneficiary_data_2["YEAR"].dt.year
#%%
# Combining state and county to a single column
combined_beneficiary_data_2["BENE_STATE_COUNTY_CODE"] = (
    combined_beneficiary_data_2["SP_STATE_CODE"].astype(str)
    + "-"
    + combined_beneficiary_data_2["BENE_COUNTY_CD"].astype(str)
)

combined_beneficiary_data_2.drop(
    columns=["SP_STATE_CODE", "BENE_COUNTY_CD"], inplace=True, axis=1
)
# %%
del (
    data_beneficiary_2008,
    data_beneficiary_2009,
    data_beneficiary_2010,
    data_inpatient_claims,
    selected_data,
    readmitted_patients_data,
    not_readmitted_patients_data,
    proc_code,
    diag_code,
    prvdr_num,
)


# %%
final_inpatient_data.columns = [
    "DESYNPUF_ID" if col == "DESYNPUF_ID" else col + "_INP"
    for col in final_inpatient_data
]

final_inpatient_data.rename(columns={"IsReadmitted_INP": "IsReadmitted"}, inplace=True)
# %%
final_df = pd.merge(
    left=combined_beneficiary_data_2,
    right=final_inpatient_data,
    left_on=["DESYNPUF_ID", "YEAR"],
    right_on=["DESYNPUF_ID", "CLAIM_YEAR_INP"],
    how="inner",
)

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
    "IsReadmitted",
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
    "YEAR",
    "CLAIM_YEAR_INP",
    "PRVDR_NUM_INP",
    "CLM_DRG_CD_INP",
    "CLM_ID_INP",
    "SEGMENT_x_INP",
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

#%%
df[categorical_features] = df[categorical_features].astype("category")


# %%
df.drop(columns=cols_to_drop + date_cols + npi_cols, inplace=True, axis=1)

#%%
# Check - 1 dropping columns as per feature importance
# df.drop(columns="BENRES_IP", inplace=True, axis=1)

# %%
X = df.drop(columns=["IsReadmitted"], axis=1)
y = df.loc[:, "IsReadmitted"]


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE_FOR_SPLIT, random_state=RANDOM_STATE
)


# %%
categorical_features = list(X.select_dtypes(include="category").columns)
numerical_features = list(X.select_dtypes(include="number").columns)


# %%
ct = StatisticalTest.ChiSquare(pd.concat([X_train, y_train], axis=1))

#%%
# # > High correlation between BENE_STATE_COUNTY_CODE & PRVDR_NUM_CAT_INP columns

# # %%
# # X_train.drop(columns=["BENE_STATE_COUNTY_CODE"], axis=1, inplace=True)
# # X_test.drop(columns=["BENE_STATE_COUNTY_CODE"], axis=1, inplace=True)

# %%
# # Very high correlation between ('BENRES_OP', 'MEDREIMB_OP') & ('BENRES_CAR', 'MEDREIMB_CAR') hence adding one of them to correlated list for dropping one column in the pair

correlated_cols_drop_list = [
    "MEDREIMB_OP",
    "MEDREIMB_CAR",
]

# %%
X_train.columns


# %%

for c in categorical_features:
    ct.TestIndependence(c, "IsReadmitted")

# %%
#%%
# Initializing all the objects for preprocessing
std_scalar = StandardScaler()
min_max_scalar = MinMaxScaler()
onehot_encoder = OneHotEncoder(drop="first", sparse=False)
median_imputer = SimpleImputer(strategy="median", missing_values=np.nan)
constant_imputer = SimpleImputer(
    strategy="constant", fill_value=MISSING_VALUE_LABEL, missing_values=np.nan
)
ordinal_encoder = OrdinalEncoder()

#%%
# Initializing custom pipelines
cardinality_reducer = CardinalityReducer(
    cutt_off=INFREQUENT_CATEGORY_CUT_OFF, label=INFREQUENT_CATEGORY_LABEL
)
column_selector = SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True)
# %%
# Temp files for caching Pipelines
numerical_cachedir = mkdtemp(prefix="num")
numerical_memory = Memory(location=numerical_cachedir, verbose=VERBOSE_PARAM_VALUE)

categorical_cachedir = mkdtemp(prefix="cat")
categorical_memory = Memory(location=categorical_cachedir, verbose=VERBOSE_PARAM_VALUE)

pipeline_cachedir = mkdtemp(prefix="pipe")
pipeline_memory = Memory(location=pipeline_cachedir, verbose=VERBOSE_PARAM_VALUE)


# %%

# Pipeline to automate the numerical column processing
numerical_transformer = Pipeline(
    steps=[("imputer_with_medium", median_imputer), ("scaler", std_scalar)],
    verbose=VERBOSE_PARAM_VALUE,
    memory=numerical_memory,
)


# %%

# Pipeline to automate the categorical column processing with onehot encoder
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
    verbose=VERBOSE_PARAM_VALUE,
    memory=categorical_memory,
)


# %%

# Pipeline to automate the categorical column processing with ordinal
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
    verbose=VERBOSE_PARAM_VALUE,
    memory=categorical_memory,
)

# %%
# Model initialization
lr = LogisticRegression(random_state=RANDOM_STATE)
lda = LinearDiscriminantAnalysis()
dt = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
rfc = RandomForestClassifier(
    bootstrap=False,
    max_features=0.440238968203593,
    n_estimators=88,
    n_jobs=N_JOB_PARAM_VALUE,
    verbose=VERBOSE_PARAM_VALUE,
    random_state=RANDOM_STATE,
)

extratrees_clf = ExtraTreesClassifier(
    max_features=0.7363742386320187,
    n_estimators=80,
    n_jobs=N_JOB_PARAM_VALUE,
    verbose=VERBOSE_PARAM_VALUE,
    random_state=RANDOM_STATE,
)

# %%
# Oversampler initialization
smt = SMOTE(random_state=RANDOM_STATE, n_jobs=N_JOB_PARAM_VALUE)
random_oversampling = RandomOverSampler(random_state=RANDOM_STATE)

#%%
# Feature selector initialization
rfe_lr = RFE(
    estimator=lr, n_features_to_select=20, verbose=VERBOSE_PARAM_VALUE, step=0.1
)
rfe_dt = RFE(
    estimator=dt, n_features_to_select=20, verbose=VERBOSE_PARAM_VALUE, step=0.1
)
rfe_rfc = RFE(
    estimator=rfc, n_features_to_select=30, verbose=VERBOSE_PARAM_VALUE, step=0.1
)
extratrees_rfc = RFE(
    estimator=extratrees_clf,
    n_features_to_select=30,
    verbose=VERBOSE_PARAM_VALUE,
    step=0.1,
)

#%%
class ModelTrainer(object):
    """
    docstring
    """

    def __init__(
        self,
        xtrain: pd.DataFrame,
        ytrain: pd.Series,
        xtest: pd.DataFrame,
        ytest: pd.Series,
        isPipeline: bool = True,
    ) -> None:
        """[summary]

        Args:
            isPipeline (bool, optional): [description]. Defaults to True.
        """
        self.isPipeline = isPipeline
        self.classifier = None
        self.X_train = xtrain
        self.y_train = ytrain
        self.X_test = xtest
        self.y_test = ytest
        self.y_pred = None
        self.y_pred_proba = None
        self.metric_df = pd.DataFrame(
            columns=[
                "Classifier_Name",
                "Accuracy",
                "Precision",
                "Recall",
                "AUC_ROC_Score",
                "F1_Score",
            ]
        )
        self.current_metrics = dict.fromkeys(
            [
                "Classifier_Name",
                "Accuracy",
                "Precision",
                "Recall",
                "AUC_ROC_Score",
                "F1_Score",
            ]
        )

    def _fit_predict(self):
        """
        docstring
        """
        self.classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.classifier.predict(X_test)
        self.y_pred_proba = self.classifier.predict_proba(X_test)

    def _get_metrics_data(self):
        self.current_metrics["Accuracy"] = accuracy_score(self.y_test, self.y_pred)
        self.current_metrics["Precision"] = precision_score(self.y_test, self.y_pred)
        self.current_metrics["Recall"] = recall_score(self.y_test, self.y_pred)
        self.current_metrics["F1_Score"] = f1_score(self.y_test, self.y_pred)
        classification_rep = classification_report(self.y_test, self.y_pred)
        self.current_metrics["AUC_ROC_Score"] = roc_auc_score(self.y_test, self.y_pred)
        (
            self.current_metrics["fpr"],
            self.current_metrics["tpr"],
            self.current_metrics["thresholds"],
        ) = roc_curve(self.y_test, self.y_pred_proba[:, 1])
        print(f"Accuracy score is {self.current_metrics['Accuracy']}")
        print(f"Precision score is {self.current_metrics['Precision']}")
        print(f"Recall score is {self.current_metrics['Recall']}")
        print(f"F1 score is {self.current_metrics['F1_Score']}")
        print(f"AUC ROC score is {self.current_metrics['AUC_ROC_Score']}")
        print(f"Classification Report is \n {classification_rep}")

    def plot_confusion_matrix_(self, axis):
        class_names = ["Not Readmitted", "Readmitted"]
        pcm = plot_confusion_matrix(
            self.classifier,
            self.X_test,
            self.y_test,
            display_labels=class_names,
            cmap="Blues",
            normalize="true",
            ax=axis,
        )

    def plot_precision_recall_curve_(self, axis):
        # Plot a simple histogram with binsize determined automatically
        prc = plot_precision_recall_curve(
            self.classifier, self.X_test, self.y_test, ax=axis
        )

    def plot_roc_curve_(self, axis):
        rac = plot_roc_curve(self.classifier, self.X_test, self.y_test, ax=axis)

    def tpr_fpr(self):
        # Evaluating model performance at various thresholds
        # The histogram of scores compared to true labels
        fig_hist = px.histogram(
            x=self.y_pred_proba[:, 1],
            color=self.y_test,
            nbins=50,
            labels=dict(color="True Labels", x="Score"),
        )

        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba[:, 1])

        df = pd.DataFrame(
            {"False Positive Rate": fpr, "True Positive Rate": tpr}, index=thresholds
        )
        df.index.name = "Thresholds"
        df.columns.name = "Rate"

        fig_thresh = px.line(
            df, title="TPR and FPR at every threshold", width=700, height=500
        )
        fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_thresh.update_xaxes(range=[0, 1], constrain="domain")
        fig_hist.show()
        fig_thresh.show()

    def plot_feature_importance(self):
        print(self.isPipeline)
        if self.isPipeline:
            feat_imp = feature_importance.FeatureImportance(self.classifier)
            feat_imp.plot(top_n_features=50, width=1000)

    def plot_curves(self):
        self.tpr_fpr()
        self.plot_feature_importance()
        fig, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=False)
        sns.despine(left=True)
        # sns.set(style="white", palette="muted", color_codes=True)
        self.plot_confusion_matrix_(axes[0, 0])
        self.plot_precision_recall_curve_(axes[0, 1])
        self.plot_roc_curve_(axes[1, 0])
        fig.delaxes(axes[1, 1])

    def add_metrics(self):
        self.current_metrics["Classifier_Name"] = (
            str(self.classifier.steps[-2:]) if self.isPipeline else str(self.classifier)
        )
        self.metric_df = self.metric_df.append(self.current_metrics, ignore_index=True)

    # apply threshold to positive probabilities to create labels
    def _get_labels_at_threshold(self, pos_probs, threshold):
        return (pos_probs >= threshold).astype("int")

    def optimize_threshold(self):
        # keep probabilities for the positive outcome only
        probs = self.y_pred_proba[:, 1]
        # define thresholds
        thresholds = np.arange(0, 1, 0.001)
        # evaluate each threshold
        scores = [
            f1_score(self.y_test, self._get_labels_at_threshold(probs, t))
            for t in thresholds
        ]
        # get best threshold
        ix = np.argmax(scores)
        precision = precision_score(
            self.y_test, self._get_labels_at_threshold(probs, thresholds[ix])
        )
        recall = recall_score(
            self.y_test, self._get_labels_at_threshold(probs, thresholds[ix])
        )
        cnf_mat = confusion_matrix(
            y_pred=self._get_labels_at_threshold(probs, thresholds[ix]),
            y_true=self.y_test,
        )
        sns.heatmap(cnf_mat, annot=True, fmt="", cmap="Blues")
        print(
            f"Threshold={thresholds[ix]:.3f}, F-Score={scores[ix]:.5f}, Precision={precision}, Recall={recall}"
        )

    def train_test_model(self, model):
        self.classifier = model
        print(f'{"="*40} {str(self.classifier)} {"="*40}')
        self._fit_predict()
        self._get_metrics_data()
        self.add_metrics()
        self.plot_curves()


#%%
# Transformer to run both the numerical and one hot encoder pipeline on specified dtypes

preprocessing_transformer = ColumnTransformer(
    [
        (
            "categorical",
            ord_categorical_transformer,
            make_column_selector(dtype_include="category"),
        ),
        (
            "numerical",
            numerical_transformer,
            make_column_selector(dtype_include=np.number),
        ),
    ],
    remainder="drop",
    verbose=VERBOSE_PARAM_VALUE,
)

# Transformer to run both the numerical and ordinal encoder pipeline on specified dtypes
# ord_preprocessing_transformer = ColumnTransformer(
#     [
#         (
#             "categorical",
#             ord_categorical_transformer,
#             make_column_selector(dtype_include="category"),
#         ),
#         (
#             "numerical",
#             numerical_transformer,
#             make_column_selector(dtype_include=np.number),
#         ),
#     ],
#     remainder="drop",
#     verbose=VERBOSE_PARAM_VALUE,
# )
# %%

# Pipeline to automate drop of specified column and running the preprocessor for the remaining column
preprocessing_pipeline = Pipeline(
    [
        ("column selector", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

#%%
imblanced_preprocessing_pipeline_smote = imb_pipeline.Pipeline(
    [
        ("column selector", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE oversampling", smt),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

imblanced_preprocessing_pipeline_randover = imb_pipeline.Pipeline(
    [
        ("column selector", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random oversampling", random_oversampling),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

#%%
X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
X_train_transformed = pd.DataFrame(
    X_train_transformed, columns=get_ct_feature_names(preprocessing_transformer)
)

#%%
X_test_transformed = preprocessing_pipeline.transform(X_test)
X_test_transformed = pd.DataFrame(
    X_test_transformed, columns=get_ct_feature_names(preprocessing_transformer)
)
#%%
(
    X_train_transformed_smote,
    y_train_transformed_smote,
) = imblanced_preprocessing_pipeline_smote.fit_resample(X_train, y_train)

#%%
X_train_transformed_smote = pd.DataFrame(
    X_train_transformed_smote, columns=get_ct_feature_names(preprocessing_transformer),
)

#%%
(
    X_train_transformed_randover,
    y_train_transformed_randover,
) = imblanced_preprocessing_pipeline_randover.fit_resample(X_train, y_train)

#%%
X_train_transformed_randover = pd.DataFrame(
    X_train_transformed_randover,
    columns=get_ct_feature_names(preprocessing_transformer),
)
#%%
pipeline_trainer = ModelTrainer(
    xtrain=X_train, ytrain=y_train, xtest=X_test, ytest=y_test,
)

ensemble_trainer = ModelTrainer(
    xtrain=X_train, ytrain=y_train, xtest=X_test, ytest=y_test, isPipeline=False
)

model_trainer = ModelTrainer(
    xtrain=X_train_transformed, ytrain=y_train, xtest=X_test_transformed, ytest=y_test
)
smote_trainer = ModelTrainer(
    xtrain=X_train_transformed_smote,
    ytrain=y_train_transformed_smote,
    xtest=X_test,
    ytest=y_test,
)
randover_trainer = ModelTrainer(
    xtrain=X_train_transformed_randover,
    ytrain=y_train_transformed_randover,
    xtest=X_test,
    ytest=y_test,
)

# %%
logreg_rfe_pipeline = Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("Feature Selection", rfe_lr),
        ("LogReg_Classifier", lr),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

logreg_rfe_smote_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE oversampling", smt),
        ("Feature Selection", rfe_lr),
        ("LogReg_Classifier", lr),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

logreg_rfe_randover_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random oversampling", random_oversampling),
        ("Feature Selection", rfe_lr),
        ("LogReg_Classifier", lr),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

# %%
logreg_pipeline = Pipeline(
    [("Preprocessing Step", preprocessing_pipeline), ("LogReg_Classifier", lr),],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

logreg_smote_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE oversampling", smt),
        ("LogReg_Classifier", lr),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

logreg_randover_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random oversampling", random_oversampling),
        ("LogReg_Classifier", lr),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)


#%%
dt_rfe_pipeline = Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("Feature Selection", rfe_dt),
        ("DT_Classifier", dt),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

dt_rfe_smote_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE oversampling", smt),
        ("Feature Selection", rfe_dt),
        ("DT_Classifier", dt),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

dt_rfe_randover_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random oversampling", random_oversampling),
        ("Feature Selection", rfe_dt),
        ("DT_Classifier", dt),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)
#%%
rfc_rfe_pipeline = Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("Feature Selection", rfe_rfc),
        ("RFC_Classifier", rfc),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

rfc_rfe_smote_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE oversampling", smt),
        ("Feature Selection", rfe_rfc),
        ("RFC_Classifier", rfc),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

rfc_rfe_randover_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random oversampling", random_oversampling),
        ("Feature Selection", rfe_rfc),
        ("RFC_Classifier", rfc),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

#%%
extratrees_rfe_pipeline = Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("Feature Selection", extratrees_rfc),
        ("extratrees_Classifier", extratrees_clf),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

extratrees_rfe_smote_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE oversampling", smt),
        ("Feature Selection", extratrees_rfc),
        ("extratrees_Classifier", extratrees_clf),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

extratrees_rfe_randover_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random oversampling", random_oversampling),
        ("Feature Selection", extratrees_rfc),
        ("extratrees_Classifier", extratrees_clf),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

# %%
dt_pipeline = Pipeline(
    [("Preprocessing Step", preprocessing_pipeline), ("DT_Classifier", dt),],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

dt_smote_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE Sampling", smt),
        ("Classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

dt_randover_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random OverSampling", random_oversampling),
        ("Classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)
#%%
rfc_pipeline = Pipeline(
    [("Preprocessing Step", preprocessing_pipeline), ("rfc_classifier", rfc),],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

rfc_smote_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE Sampling", smt),
        ("rfc_classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

rfc_randover_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random OverSampling", random_oversampling),
        ("rfc_classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)
#%%
extratrees_pipeline = Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("extratrees_classifier", extratrees_clf),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

extratrees_smote_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE Sampling", smt),
        ("extratrees_classifier", extratrees_clf),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

extratrees_randover_pipeline = imb_pipeline.Pipeline(
    [
        ("drop columns", column_selector,),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random OverSampling", random_oversampling),
        ("extratrees_classifier", extratrees_clf),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

#%%
pipeline_trainer.train_test_model(logreg_rfe_smote_pipeline)

#%%
pipeline_trainer.train_test_model(logreg_rfe_pipeline)
#%%
pipeline_trainer.train_test_model(logreg_rfe_randover_pipeline)

#%%
pipeline_trainer.train_test_model(dt_rfe_smote_pipeline)

#%%
pipeline_trainer.train_test_model(dt_rfe_pipeline)
#%%
pipeline_trainer.train_test_model(dt_rfe_randover_pipeline)

#%%
pipeline_trainer.train_test_model(rfc_rfe_smote_pipeline)

#%%
pipeline_trainer.train_test_model(rfc_rfe_pipeline)
#%%
pipeline_trainer.train_test_model(rfc_rfe_randover_pipeline)


#%%
pipeline_trainer.train_test_model(logreg_smote_pipeline)

#%%
pipeline_trainer.train_test_model(logreg_pipeline)
#%%
pipeline_trainer.train_test_model(logreg_randover_pipeline)

#%%
pipeline_trainer.train_test_model(dt_smote_pipeline)

#%%
pipeline_trainer.train_test_model(dt_pipeline)
#%%
pipeline_trainer.train_test_model(dt_randover_pipeline)

#%%
pipeline_trainer.train_test_model(rfc_smote_pipeline)

#%%
pipeline_trainer.train_test_model(rfc_pipeline)
#%%
pipeline_trainer.train_test_model(rfc_randover_pipeline)

#%%
pipeline_trainer.train_test_model(extratrees_smote_pipeline)

#%%
pipeline_trainer.train_test_model(extratrees_pipeline)
#%%
pipeline_trainer.train_test_model(extratrees_randover_pipeline)
#%%
pipeline_trainer.train_test_model(extratrees_rfe_smote_pipeline)

#%%
pipeline_trainer.train_test_model(extratrees_rfe_pipeline)
#%%
pipeline_trainer.train_test_model(extratrees_rfe_randover_pipeline)

#%%
estimators = [
    ("extratrees_pipeline", extratrees_pipeline),
    ("rfc_pipeline", rfc_pipeline),
]
voting_clf = VotingClassifier(
    estimators=estimators,
    voting="soft",
    n_jobs=N_JOB_PARAM_VALUE,
    verbose=VERBOSE_PARAM_VALUE,
)
ensemble_trainer.train_test_model(voting_clf)
#%%
# dump(
#     rfc_rfe_randover_pipeline,
#     f"{MODEL_REPOSITORY_LOCATION}\\rfc_rfe_randover_pipeline.pkl",
# )
# # %%
# dump(
#     X_train.columns.tolist(),
#     f"{MODEL_REPOSITORY_LOCATION}\\rfc_rfe_randover_pipeline_cols.pkl",
# )
#%%
# Initialize clustering objects
kmeans_clustering = KMeans(
    n_clusters=5,
    verbose=VERBOSE_PARAM_VALUE,
    init="k-means++",
    max_iter=300,
    random_state=RANDOM_STATE,
    n_init=10,
)

mini_batch_kmeans = MiniBatchKMeans(
    n_clusters=2,
    verbose=VERBOSE_PARAM_VALUE,
    random_state=RANDOM_STATE,
    batch_size=20000,
    max_iter=10,
)

agglomeratrive_clustering = AgglomerativeClustering()
#%%
# wcss = []

# for i in range(1, 25):
#     km = KMeans(
#         n_clusters=i,
#         init="k-means++",
#         max_iter=500,
#         n_init=20,
#         random_state=0,
#         verbose=VERBOSE_PARAM_VALUE,
#     )
#     print(f"Fitting started for {i}")
#     km.fit(X_train_transformed)
#     print(f"Fitting completed for {i}")
#     wcss.append(km.inertia_)
# #%%

# plt.figure(figsize=(20, 20))
# sns.lineplot(x=range(1, 25), y=wcss, marker="X", linewidth=2, markersize=12)
# plt.show()

# %%


famd = FAMD(n_components=2, n_iter=10, random_state=RANDOM_STATE)
famd.fit(X=X_train_transformed)
X_train_transformed_famd = famd.transform(X_train_transformed)
# %%
pca = PCA(n_components=2, random_state=RANDOM_STATE,)
pca.fit(X_train_transformed)


# %%
X_train_transformed_pca = pca.transform(X_train_transformed)
# %%
