#%%
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# %%
import pandas as pd
import numpy as np

#%%
import json

#%%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn import pipeline as imb_pipeline

# %%
# Custom packages
from utils import (
    ExtractData,
    feature_importance,
    StatisticalTest,
    CategorizeCardinalData,
    CustomPipeline,
)
from src import dispatcher

#%%
# loading configuration values
with open("../config.json", "r") as f:
    config = json.load(f)

#%%
NON_NAN_THRESHOLD = config[0]["threshold_nan"]
TEST_SIZE_FOR_SPLIT = config[3]["TEST_SIZE_FOR_SPLIT"]
RANDOM_STATE = config[3]["RANDOM_STATE"]

# %%
data = ExtractData.FetchSubset(no_of_subset=1)

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
    )["Diagnosis_code_CAT"]

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
    data_inpatient_claims[diagnosis_code_cat].isin([selected_icd_code]).any(axis=1), :
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
selected_data.loc[:, "Readmission_within_30days"] = (
    selected_data.loc[:, "Readmission_Day_Count"] <= 30
) * 1

selected_data.drop(columns="Readmission_Day_Count", inplace=True)
selected_data.sort_values(
    by=["DESYNPUF_ID", "NCH_BENE_DSCHRG_DT"], ascending=True, inplace=True
)
#%%
# Removing the not readmitted record for the patient which have atleast one admitted record
readmitted_patients_data = selected_data.loc[
    (selected_data["Readmission_within_30days"] == 1), :
]
readmitted_patients = readmitted_patients_data.loc[:, "DESYNPUF_ID"].unique()

drop_index = selected_data.loc[
    (selected_data["DESYNPUF_ID"].isin(readmitted_patients))
    & (selected_data["Readmission_within_30days"] == 0)
].index

selected_data.drop(index=drop_index, axis=0, inplace=True)
readmitted_patients_data = selected_data.loc[
    (selected_data["Readmission_within_30days"] == 1), :
]
not_readmitted_patients_data = selected_data.loc[
    (selected_data["Readmission_within_30days"] == 0), :
]

# Uncomment the below line to remove duplicate patient from non readmitted data
# not_readmitted_patients_data.drop_duplicates(subset=["DESYNPUF_ID", "Readmission_within_30days"], keep='last', inplace=True)

final_inpatient_data = pd.concat(
    [readmitted_patients_data, not_readmitted_patients_data], axis=0
)

#%%
final_inpatient_data.drop(columns=icd_procedural_code + diagnosis_code, inplace=True)

# %%
data_beneficiary_2008 = data.fetchFromBeneficiaryDataset(year=2008)
data_beneficiary_2008["YEAR"] = pd.to_datetime("2008-12-31", infer_datetime_format=True)
data_beneficiary_2009 = data.fetchFromBeneficiaryDataset(year=2009)
data_beneficiary_2009["YEAR"] = pd.to_datetime("2009-12-31", infer_datetime_format=True)
data_beneficiary_2010 = data.fetchFromBeneficiaryDataset(year=2010)
data_beneficiary_2010["YEAR"] = pd.to_datetime("2010-12-31", infer_datetime_format=True)

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
    "YEAR",
    "CLAIM_YEAR_INP",
    "PRVDR_NUM_INP",
    "CLM_DRG_CD_INP",
    "CLM_ID_INP",
    "SEGMENT_INP",
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
X = df.drop(columns=["Readmission_within_30days_INP"], axis=1)
y = df.loc[:, "Readmission_within_30days_INP"]


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
# # Very high correlation between ('BENRES_OP', 'MEDREIMB_OP') & ('BENRES_CAR', 'MEDREIMB_CAR') hence dropping one column in the pair

# X_train.drop(columns=["MEDREIMB_OP", "MEDREIMB_CAR"], inplace=True)
# X_test.drop(columns=["MEDREIMB_OP", "MEDREIMB_CAR"], inplace=True)
correlated_cols_drop_list = ["MEDREIMB_OP", "MEDREIMB_CAR"]

# %%
X_train.columns


# %%

for c in categorical_features:
    ct.TestIndependence(c, "Readmission_within_30days_INP")

# %%
categorical_features = list(X_train.select_dtypes(include="category").columns)
numerical_feature = list(X_train.select_dtypes(include="number").columns)

# %%
def fit_predict(pipeline: Pipeline):
    model = pipeline.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{"="*40} {pipeline.steps[-1][-1]} {"="*50}')
    print(f'{"="*40} {"Training Metrics"} {"="*50}')
    dispatcher.getMetricsData(y_train, model.predict(X_train))
    print(f'{"="*40} {"Testing Metrics"} {"="*50}')
    dispatcher.getMetricsData(y_test, y_pred)


#%%
dispatcher.logreg_rfe_pipeline.steps.insert(
    0,
    (
        "column selector",
        CustomPipeline.SelectColumnsTransfomer(
            columns=correlated_cols_drop_list, drop=True
        ),
    ),
)
#%%
fit_predict(dispatcher.logreg_rfe_pipeline)
#%%
dispatcher.logreg_rfe_pipeline.predict(X_test)

#%%
dispatcher.logreg_rfe_smote_pipeline.fit(X_train, y_train)
#%%
dispatcher.dt_rfe_pipeline.fit(X_train, y_train)
dispatcher.rfc_rfe_pipeline.fit(X_train, y_train)
dispatcher.dt_rfe_smote_pipeline.fit(X_train, y_train)
dispatcher.rfc_rfe_smote_pipeline.fit(X_train, y_train)

# %%
