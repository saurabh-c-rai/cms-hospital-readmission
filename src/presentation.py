# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CMS Readmission Analysis
#
# Approximately 4.3 million hospital readmissions occur each year in the U.S., costing more than \\$ 60 billion, with preventable adverse patient events creating additional clinical and financial burdens for both patients and healthcare systems.
#
# Total Medicare penalties assessed on hospitals for readmissions increased to \\$ 528 million in 2017, \\$ 108 million more than in 2016. The increase is due mostly to more medical conditions being measured. Hospital fines will average less than 1 percent of their Medicare inpatient payments.

# %% [markdown]
# ## Problem Statement
#
# 1. Use AI methodologies to predict unplanned hospital and Skilled Nursing
# Facilities admissions and adverse events within 30 days for Medicare
# beneficiaries, based on a data set of Medicare administrative claims data,
# including Medicare Part A (hospital) and Medicare Part B (professional
# services).
# 2. Develop innovative strategies and methodologies to: explain the AI-derived
# predictions to front-line clinicians and patients to aid in providing appropriate
# clinical resources to model participants

# %% [markdown]
# ## About the dataset
#
# The dataset is divided into 20 samples. Each samples contains below zipped csv files :-
#
# - Beneficiary_Summary_File_Sample_1 : 3 files containing beneficiary summary for 2008, 2009 & 2010
# - 2008_to_2010_Inpatient_Claims_Sample_1 : files containing inpatient claim details for 2008-2010
# - 2008_to_2010_Outpatient_Claims_Sample_1 : files containing outpatient claim details for 2008-2010
# - 2008_to_2010_Carrier_Claims_Sample_1 : 2 files containing carrier claim details of A & B for 2008-2010
# - Prescription_Drug_Events_Sample_1 : file containing drug events for 2008-2009

# %% [markdown]
# The CMS Beneficiary Summary DE-SynPUF contains 32 variables. Each record pertains to a
# synthetic Medicare beneficiary and contains:
#
# |# |Variable names | Labels |
# |:----|:-----|:-----|
# | 1 | DESYNPUF_ID | DESYNPUF: Beneficiary Code|
# | 2 | BENE_BIRTH_DT | DESYNPUF: Date of birth|
# | 3 | BENE_DEATH_DT | DESYNPUF: Date of death|
# | 4 | BENE_SEX_IDENT_CD | DESYNPUF: Sex|
# | 5 | BENE_RACE_CD | DESYNPUF: Beneficiary Race Code|
# | 6 | BENE_ESRD_IND | DESYNPUF: End stage renal disease Indicator|
# | 7 | SP_STATE_CODE | DESYNPUF: State Code|
# | 8 | BENE_COUNTY_CD | DESYNPUF: County Code|
# | 9 | BENE_HI_CVRAGE_TOT_MONS | DESYNPUF: Total number of months of part A coverage for the beneficiary.|
# | 10 | BENE_SMI_CVRAGE_TOT_MONS | DESYNPUF: Total number of months of part B coverage for the beneficiary.|
# | 11 | BENE_HMO_CVRAGE_TOT_MONS | DESYNPUF: Total number of months of HMO coverage for the beneficiary.|
# | 12 | PLAN_CVRG_MOS_NUM | DESYNPUF: Total number of months of part D plan coverage for the beneficiary.|
# | 13 | SP_ALZHDMTA | DESYNPUF: Chronic Condition: Alzheimer or related disorders or senile|
# | 14 | SP_CHF | DESYNPUF: Chronic Condition: Heart Failure|
# | 15 | SP_CHRNKIDN | DESYNPUF: Chronic Condition: Chronic Kidney Disease|
# | 16 | SP_CNCR | DESYNPUF: Chronic Condition: Cancer|
# | 17 | SP_COPD | DESYNPUF: Chronic Condition: Chronic Obstructive Pulmonary Disease|
# | 18 | SP_DEPRESSN | DESYNPUF: Chronic Condition: Depression|
# | 19 | SP_DIABETES | DESYNPUF: Chronic Condition: Diabetes|
# | 20 | SP_ISCHMCHT | DESYNPUF: Chronic Condition: Ischemic Heart Disease|
# | 21 | SP_OSTEOPRS | DESYNPUF: Chronic Condition: Osteoporosis|
# | 22 | SP_RA_OA | DESYNPUF: Chronic Condition: rheumatoid arthritis and osteoarthritis (RA/OA)|
# | 23 | SP_STRKETIA | DESYNPUF: Chronic Condition: Stroke/transient Ischemic Attack|
# | 24 | MEDREIMB_IP | DESYNPUF: Inpatient annual Medicare reimbursement amount|
# | 25 | BENRES_IP | DESYNPUF: Inpatient annual beneficiary responsibility amount|
# | 26 | PPPYMT_IP | DESYNPUF: Inpatient annual primary payer reimbursement amount|
# | 27 | MEDREIMB_OP | DESYNPUF: Outpatient Institutional annual Medicare reimbursement amount|
# | 28 | BENRES_OP | DESYNPUF: Outpatient Institutional annual beneficiary responsibility amount|
# | 29 | PPPYMT_OP | DESYNPUF: Outpatient Institutional annual primary payer reimbursement amount|
# | 30 | MEDREIMB_CAR | DESYNPUF: Carrier annual Medicare reimbursement amount|
# | 31 | BENRES_CAR | DESYNPUF: Carrier annual beneficiary responsibility amount|
# | 32 | PPPYMT_CAR | DESYNPUF: Carrier annual primary payer reimbursement amount                                     |

# %% [markdown]
# The CMS Inpatient Claims DE-SynPUF contains 81 variables. Each record pertains to a synthetic
# inpatient claim and contains:
#
# |# |Variable names | Labels |
# |:----|:-----|:-----|
# | 1 | DESYNPUF_ID |DESYNPUF: Beneficiary Code|
# | 2 | CLM_ID |DESYNPUF: Claim ID|
# | 3 | SEGMENT |DESYNPUF: Claim Line Segment|
# | 4 | CLM_FROM_DT |DESYNPUF: Claims start date|
# | 5 | CLM_THRU_DT |DESYNPUF: Claims end date|
# | 6 | PRVDR_NUM |DESYNPUF: Provider Institution|
# | 7 | CLM_PMT_AMT |DESYNPUF: Claim Payment Amount|
# | 8 | NCH_PRMRY_PYR_CLM_PD_AMT |DESYNPUF: NCH Primary Payer Claim Paid Amount|
# | 9 | AT_PHYSN_NPI |DESYNPUF: Attending Physician – National Provider Identifier Number|
# | 10| OP_PHYSN_NPI |DESYNPUF: Operating Physician – National Provider Identifier Number|
# | 11| OT_PHYSN_NPI |DESYNPUF: Other Physician – National Provider Identifier Number|
# | 12| CLM_ADMSN_DT | DESYNPUF: Inpatient admission date|
# | 13| ADMTNG_ICD9_DGNS_CD | DESYNPUF: Claim Admitting Diagnosis Code|
# | 14| CLM_PASS_THRU_PER_DIEM_AMT | DESYNPUF: Claim Pass Thru Per Diem Amount|
# | 15| NCH_BENE_IP_DDCTBL_AMT | DESYNPUF: NCH Beneficiary Inpatient Deductible Amount|
# | 16| NCH_BENE_PTA_COINSRNC_LBLTY_AM | DESYNPUF: NCH Beneficiary Part A Coinsurance Liability Amount|
# | 17| NCH_BENE_BLOOD_DDCTBL_LBLTY_AM | DESYNPUF: NCH Beneficiary Blood Deductible Liability Amount|
# | 18| CLM_UTLZTN_DAY_CNT | DESYNPUF: Claim Utilization Day Count|
# | 19| NCH_BENE_DSCHRG_DT | DESYNPUF: Inpatient discharged date|
# | 20| CLM_DRG_CD | DESYNPUF: Claim Diagnosis Related Group Code|
# | 21-30| ICD9_DGNS_CD_1 – ICD9_DGNS_CD_10 | DESYNPUF: Claim Diagnosis Code 1 – Claim Diagnosis Code 10|
# | 31-36|ICD9_PRCDR_CD_1 – ICD9_PRCDR_CD_6 | DESYNPUF: Claim Procedure Code 1 – Claim Procedure Code 6|
# |37-81| HCPCS_CD_1 – HCPCS_CD_45 | DESYNPUF: Revenue Center HCFA Common Procedure Coding System 1 – Revenue Center HCFA Common Procedure Coding System 45|

# %% [markdown]
# The CMS Outpatient Claims DE-SynPUF contains 76 variables. Each record pertains to a synthetic
# outpatient claim and contains:
#
# |# |Variable names | Labels |
# |:----|:-----|:-----|
# | 1 | DESYNPUF_ID | DESYNPUF: Beneficiary Code|
# | 2 | CLM_ID | DESYNPUF: Claim ID|
# | 3 | SEGMENT | DESYNPUF: Claim Line Segment|
# | 4 | CLM_FROM_DT | DESYNPUF: Claims start date|
# | 5 | CLM_THRU_DT | DESYNPUF: Claims end date|
# | 6 | PRVDR_NUM | DESYNPUF: Provider Institution|
# | 7 | CLM_PMT_AMT | DESYNPUF: Claim Payment Amount|
# | 8 | NCH_PRMRY_PYR_CLM_PD_AMT | DESYNPUF: NCH Primary Payer Claim Paid Amount|
# | 9 | AT_PHYSN_NPI | DESYNPUF: Attending Physician – National Provider Identifier Number|
# | 10| OP_PHYSN_NPI | DESYNPUF: Operating Physician – National Provider Identifier Number|
# | 11| OT_PHYSN_NPI | DESYNPUF: Other Physician – National Provider Identifier Number|
# | 12| NCH_BENE_BLOOD_DDCTBL_LBLTY_AM | DESYNPUF: NCH Beneficiary Blood Deductible Liability Amount|
# | 13 -22| ICD9_DGNS_CD_1 – ICD9_DGNS_CD_10 | DESYNPUF: Claim Diagnosis Code 1 – Claim Diagnosis Code 10|
# | 23 -28| ICD9_PRCDR_CD_1 – ICD9_PRCDR_CD_6 | DESYNPUF: Claim Procedure Code 1 – Claim Procedure Code 6|
# | 29| NCH_BENE_PTB_DDCTBL_AMT | DESYNPUF: NCH Beneficiary Part B Deductible Amount|
# | 30| NCH_BENE_PTB_COINSRNC_AMT | DESYNPUF: NCH Beneficiary Part B Coinsurance Amount|
# | 31| ADMTNG_ICD9_DGNS_CD | DESYNPUF: Claim Admitting Diagnosis Code|
# | 32 -76| HCPCS_CD_1 – HCPCS_CD_45| DESYNPUF: Revenue Center HCFA Common Procedure Coding System 1 – Revenue Center HCFA Common Procedure Coding System 45|

# %% [markdown]
# The CMS Carrier Claims DE-SynPUF contains 142 variables. Each record pertains to a synthetic
# physician/supplier claim and contains:
#
# |# |Variable names | Labels |
# |:----|:-----|:-----|
# | 1 | DESYNPUF_ID | DESYNPUF: Beneficiary Code|
# | 2 | CLM_ID | DESYNPUF: Claim ID|
# | 3 | CLM_FROM_DT | DESYNPUF: Claims start date|
# | 4 | CLM_THRU_DT | DESYNPUF: Claims end date|
# | 5-12 | ICD9_DGNS_CD_1 – ICD9_DGNS_CD_8 | DESYNPUF: Claim Diagnosis Code 1 – Claim Diagnosis Code 8|
# | 13-25|  PRF_PHYSN_NPI_1 – PRF_PHYSN_NPI_13 | DESYNPUF: Provider Physician – National Provider Identifier Number|
# | 26-38|  TAX_NUM_1 – TAX_NUM_13 | DESYNPUF: Provider Institution Tax Number|
# | 39-51|  HCPCS_CD_1 – HCPCS_CD_13 | DESYNPUF: Line HCFA Common Procedure Coding System 1 – Line HCFA Common Procedure Coding System 13|
# | 52-64|  LINE_NCH_PMT_AMT_1 – LINE_NCH_PMT_AMT_13 | DESYNPUF: Line NCH Payment Amount 1 – Line NCH Payment Amount 13|
# | 65-77|  LINE_BENE_PTB_DDCTBL_AMT_1 – LINE_BENE_PTB_DDCTBL_AMT_13 | DESYNPUF: Line Beneficiary Part B Deductible Amount 1 – Line Beneficiary Part B Deductible Amount 13|
# | 78-90|  LINE_BENE_PRMRY_PYR_PD_AMT_1 – LINE_BENE_PRMRY_PYR_PD_AMT_13 | DESYNPUF: Line Beneficiary Primary Payer Paid Amount 1 – Line Beneficiary Primary Payer Paid Amount 13|
# | 91-103 |  LINE_COINSRNC_AMT_1 – LINE_COINSRNC_AMT_13 | DESYNPUF: Line Coinsurance Amount 1 – Line Coinsurance Amount 13|
# | 104-116|  LINE_ALOWD_CHRG_AMT_1 – LINE_ALOWD_CHRG_AMT_13 | DESYNPUF: Line Allowed Charge Amount 1 – Line Allowed Charge Amount 13|
# | 117-129|  LINE_PRCSG_IND_CD_1 – LINE_PRCSG_IND_CD_13 | DESYNPUF: Line Processing Indicator Code 1 – Line Processing Indicator Code13|
# |130-142| LINE_ICD9_DGNS_CD_1 – LINE_ICD9_DGNS_CD_13 | DESYNPUF: Line Diagnosis Code 1 – Line Diagnosis Code 13|

# %% [markdown]
# The CMS Prescription Drug Events (PDE) DE-SynPUF contains 8 variables. Each record pertains to
# a synthetic Part D event and contains:
#
# |# |Variable names | Labels |
# |:----|:-----|:-----|
# | 1 | DESYNPUF_ID | DESYNPUF: Beneficiary Code|
# | 2 | PDE_ID | DESYNPUF: CCW Part D Event Number|
# | 3 | SRVC_DT | DESYNPUF: RX Service Date|
# | 4 | PROD_SRVC_ID | DESYNPUF: Product Service ID|
# | 5 | QTY_DSPNSD_NUM | DESYNPUF: Quantity Dispensed|
# | 6 | DAYS_SUPLY_NUM | DESYNPUF: Days Supply|
# | 7 | PTNT_PAY_AMT | DESYNPUF: Patient Pay Amount|
# | 8 | TOT_RX_CST_AMT | DESYNPUF: Gross Drug Cost|

# %% [markdown]
# # Loading libraries and configuration

# %% [markdown]
# ### Importing libraries

# %%
# To print output of all the lines
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# For processing json and storing transform result
import json
import os
from tempfile import mkdtemp

# Temporary data storage
import sqlite3

# cd_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(cd_path)

# Computation packages
import numpy as np
import pandas as pd

# Imbalanced class handling packages
from imblearn import pipeline as imb_pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler


# Pipelines
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config

# Custom packages
from utils import (
    CategorizeCardinalData,
    ExtractData,
    StatisticalTest,
    feature_importance,
)
from utils.CustomPipeline import (
    CardinalityReducer,
    get_ct_feature_names,
    SelectColumnsTransfomer,
    CategoricalVariableImputer
)


# Plotting packages
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px


from joblib import Memory, dump, load


# Pipelines & transformers
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline


# module for dimensionality reduction
from sklearn.decomposition import PCA, IncrementalPCA


# Preprocessing modules
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


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


# clustering
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN

# Statistics packages
from scipy import stats as ss
import statsmodels.api as sm

# Garbage collection
import gc

# %% [markdown]
# ### Loading configuration file

# %%
# loading configuration values
with open("../config.json", "r") as f:
    config = json.load(f)


# %%
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


# %% [markdown]
# # Section 1 : Problem Statements and Hypothesis 

# %% [markdown]
# ### L1  Articulate and create workflow of your understanding and approach to problem given. It should clearly mention the roadmap of how you will build and provide insights to stakeholders.

# %% [markdown]
# - Get an understanding of the problem statement and define it terms of a machine learning problem
# - Load a sample of the dataset & perform EDA.
# - Merge the inpatient and beneficiary data to combine the hospital admission details with the beneficiary details.
# - Generate the target variable by comparing the difference between discharge and readmission date.
# - Form hypothesis whether the readmission rate is related to race of the patient and verify if it holds on the sample.
# - Check the correlation with the groups of numerical variables & categorical variables.
# - Perform a chi2 test on the categorical variables to test if its useful for prediction of target.
# - Run a classification algorithm to get a baseline score.
# - Improve the score by using feature engineering and selecting different models.
# - Expose the model as a flask API.
# - Share the insights from plotly - dash dashboard
# - Create model using all the data on GCP.

# %% [markdown]
# ### Hypothesis Testing : Formulate 1 hypothesis on this sample data which you would like to test/potentially beneficial to know for targeted stakeholders to validate your solution

# %% [markdown]
# #### Loading dataset

# %%
conn_object = sqlite3.connect(f'{config[1]["database_path"]}')
final_data = pd.read_sql_query("SELECT * FROM final_readmission_df", con=conn_object)

# %% [markdown]
# #### Chi2 Hypothesis Test

# %% [markdown]
# > Hypothesis 1 : Race and patient gender are related
#
# $$
# H_{0} : \ BENE\_RACE\_CD \ and \ BENE\_SEX\_IDENT\_CD \ are \ not \ related\\
# H_{a} : \ BENE\_RACE\_CD \ and \ BENE\_SEX\_IDENT\_CD \ are \ related\\
# $$

# %%
Observed_df = pd.crosstab(final_data['BENE_SEX_IDENT_CD'], final_data['BENE_RACE_CD'])

# %%
chi2, p, dof, expected = ss.chi2_contingency(observed=Observed_df)

# %%
print(f"Chi2 statistics is {chi2} & p value is {p}")

# %% [markdown]
# $$
# pvalue >= 0.05 --> Failed \ to \ Reject \ H_{0}
# $$

# %% [markdown]
# > Hypothesis 2 : Race and patient gender are related
#
# $$
# H_{0} : \ Readmission\_within\_30days\_INP \ and \ BENE\_SEX\_IDENT\_CD \ are \ not \ related\\
# H_{a} : \ Readmission\_within\_30days\_INP \ and \ BENE\_SEX\_IDENT\_CD \ are \ related\\
# $$

# %%
Observed_df = pd.crosstab(final_data['BENE_SEX_IDENT_CD'], final_data['IsReadmitted'])

# %%
chi2, p, dof, expected = ss.chi2_contingency(observed=Observed_df)

# %%
print(f"Chi2 statistics is {chi2} & p value is {p}")

# %% [markdown]
# $$
# pvalue >= 0.05 --> Failed \ to \ Reject \ H_{0}
# $$

# %% [markdown]
# ### Provide Summary Statistics and inferences about data using statistics.

# %%
final_data.describe()

# %%
# import gc
# del final_data
# gc.collect()

# %% [markdown]
# > For details : Check dashboard

# %% [markdown]
# ### Create R based dashboard (ie. Shiny R dashboard) with data insights , patient timeline, 3-4 different key metrics. Dashboard should have actionable conclusions , not informative metrics.

# %% [markdown]
# > Created dashboard using dash in python

# %% [markdown]
# # Section 3: Feature Engineering & Insights

# %%
data = ExtractData.FetchSubset(subset_list=[2])

# %% [markdown]
# ## L1. Identifying Beneficiaries Enrolled in Different Time Periods (Points 10)

# %%
# %%bigquery --project dsapac  --use_bqstorage_api final_df

SELECT * FROM dsapac.CMS.

# %%
data_beneficary_summary.loc[data_beneficary_summary['Year'] == 2008, "DESYNPUF_ID"]
data_beneficary_summary.loc[data_beneficary_summary['Year'] == 2009, "DESYNPUF_ID"]
data_beneficary_summary.loc[data_beneficary_summary['Year'] == 2010, "DESYNPUF_ID"]

# %% [markdown]
# ## L2. Find beneficiaries who enrolled in all three years and had at least one inpatient claim from 2008 to 2010

# %%
# %%bigquery --project dsapac  --use_bqstorage_api requiredBeneficiary

SELECT * FROM dsapac.CMS.BeneficiariesWithOneClaim

# %%
requiredBeneficiary

# %% [markdown]
# ## L3. Number of Claims per Beneficiary by Service Type Over Three Years

# %%
beneficiary_by_service_type.loc[beneficiary_by_service_type['YearOfAdmission']==2010, ['InpatientClaims','OutpatientClaims', 'CarrierClaims']].melt().groupby("variable").agg({"value":"sum"})

# %%
beneficiary_by_service_type.groupby(["DESYNPUF_ID", "YearOfAdmission"]).sum()

# %%
# del data_carrier_claim_A
# gc.collect()

# %% [markdown]
# ## L4. Create following data tables with required formatting as follows 

# %%
# %%bigquery --project dsapac  --use_bqstorage_api demo_dist

SELECT * FROM dsapac.CMS.Demography_Distribution

# %% [markdown]
# ### Distribution by Birth Year

# %%
demo_dist.loc[ :, ['YearofAdmission','YearSpan', 'PercentBirthYear']].drop_duplicates().sort_values(by=["YearofAdmission", "YearSpan"])

# %% [markdown]
# ### Distribution by Gender

# %%
demo_dist.loc[ :, ["BENE_SEX_IDENT_CD", "PercentSex"]].drop_duplicates()

# %% [markdown]
# ### Distribution by Race

# %%
demo_dist.loc[ :, ["YearofAdmission", "BENE_RACE_CD", "PercentRace"]].sort_values(by=["YearofAdmission"]).drop_duplicates(subset=["BENE_RACE_CD", "PercentRace"])

# %%
# %%bigquery --project dsapac  --use_bqstorage_api demo_dist

SELECT * FROM dsapac.CMS.Demography_Distribution

#### Distribution by Birth Year

demo_dist.loc[ :, ['YearofAdmission','YearSpan', 'PercentBirthYear']].drop_duplicates().sort_values(by=["YearofAdmission", "YearSpan"])

#### Distribution by Gender

demo_dist.loc[ :, ["BENE_SEX_IDENT_CD", "PercentSex"]].drop_duplicates()

### Distribution by Race

demo_dist.loc[ :, ["YearofAdmission", "BENE_RACE_CD", "PercentRace"]].sort_values(by=["YearofAdmission"]).drop_duplicates(subset=["BENE_RACE_CD", "PercentRace"])

# %% [markdown]
# ### Reimbursement by Source Year

# %%
# %%bigquery --project dsapac  --use_bqstorage_api reimbursement_details

SELECT * FROM dsapac.CMS.ReimbursementSourceYear

# %%
reimbursement_details

# %% [markdown]
# # Data Processing

# %% [markdown]
# ## Processing of InPatientClaims

# %% [markdown]
# ### Dealing with missing values
# Lets check the number of missing values in the given dataset

# %%
# %%bigquery --project dsapac  --use_bqstorage_api data_inpatient_claims

SELECT * FROM dsapac.CMS.Inpatient

# %%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    data_inpatient_claims.isna().sum()[data_inpatient_claims.isna().sum() > 0]/data_inpatient_claims.shape[0]

# %% [markdown]
# ### Remove the columns which contains more than 60% of missing values

# %%
not_nan_threshold = config[0]['threshold_nan']

# %%
data_inpatient_claims.dropna(thresh=data_inpatient_claims.shape[0]*not_nan_threshold, how='all', axis=1, inplace=True)

# %%
#columns that are remaining
data_inpatient_claims.columns

# %% [markdown]
# ### Correct the dtypes

# %%
data_inpatient_claims.dtypes

# %% [markdown]
# > 'DESYNPUF_ID', 'CLM_ID', 'SEGMENT', 'AT_PHYSN_NPI', 'OP_PHYSN_NPI' are identity columns and hence changing them to objects
#
# > Creating a list of categorical columns and changing them to categories
#
# > Remaining are the numerical columns

# %%
data_inpatient_claims[['DESYNPUF_ID', 'CLM_ID', 'SEGMENT', 'AT_PHYSN_NPI', 'OP_PHYSN_NPI']] = data_inpatient_claims[['DESYNPUF_ID', 'CLM_ID', 'SEGMENT', 'AT_PHYSN_NPI', 'OP_PHYSN_NPI']].astype("object")

# %%
data_inpatient_claims['CLM_FROM_DT'] = pd.to_datetime(data_inpatient_claims['CLM_FROM_DT'], format="%Y%m%d")
data_inpatient_claims['CLM_THRU_DT'] = pd.to_datetime(data_inpatient_claims['CLM_THRU_DT'], format="%Y%m%d")
data_inpatient_claims['CLM_ADMSN_DT'] = pd.to_datetime(data_inpatient_claims['CLM_ADMSN_DT'], format="%Y%m%d")
data_inpatient_claims['NCH_BENE_DSCHRG_DT'] = pd.to_datetime(data_inpatient_claims['NCH_BENE_DSCHRG_DT'], format="%Y%m%d")

# %%
categorical_columns = ['PRVDR_NUM', 'CLM_DRG_CD', 'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6', 'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8', 'ICD9_DGNS_CD_9', 'ICD9_PRCDR_CD_1']

# %%
data_inpatient_claims[categorical_columns] = data_inpatient_claims[categorical_columns].astype('category')

# %%
numerical_columns = data_inpatient_claims.select_dtypes(exclude=["category", "object", "datetime"]).columns

# %% [markdown]
# ### Create a year column for merging with beneficiary summary data

# %%

data_inpatient_claims["CLAIM_YEAR"] = data_inpatient_claims["CLM_ADMSN_DT"].dt.year

# %% [markdown]
# ### Processing and Categorizing PROVIDER NUM

# %%
# Processing of PRVDR_NUM column. Converting the data into categories
data_inpatient_claims["PRVDR_NUM"].value_counts()
prvdr_num = CategorizeCardinalData.ProviderNumCategoryCreator()
prvdr_num.get_categories_for_providers(data_inpatient_claims["PRVDR_NUM"])
data_inpatient_claims = data_inpatient_claims.merge(
    right=prvdr_num.unique_prvdr_num_category_df, on=["PRVDR_NUM"], how="left"
)

# %% [markdown]
# ### Processing of ICD related columns

# %%
diagnosis_code = [col for col in data_inpatient_claims.columns if "ICD9_DGNS" in col]
icd_procedural_code = [
    col for col in data_inpatient_claims.columns if "ICD9_PRCDR" in col
]
icd_hcpcs_code = [col for col in data_inpatient_claims.columns if "HCPCS_CD" in col]


# %%
# Preparing dataframe for categorizing
data_inpatient_claims[diagnosis_code + icd_procedural_code + icd_hcpcs_code] = (
    data_inpatient_claims[diagnosis_code + icd_procedural_code + icd_hcpcs_code]
    .astype("str")
    .replace(["nan", "na"], np.nan)
)


# %% [markdown]
# #### Processing of Procdural Code columns

# %%
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


# %% [markdown]
# #### Processing of Diagnosis Code

# %%
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
fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(15, 20))
plt.subplots_adjust(hspace=1.8)
for i, ax in enumerate(fig.axes):
    subset_data = data_inpatient_claims[["DESYNPUF_ID", f"{diagnosis_code[i]}_CAT"]]
    series_plot = subset_data.groupby([f"{diagnosis_code[i]}_CAT"])["DESYNPUF_ID"].size()
    plot = sns.barplot(x=series_plot.index, y=series_plot.values, ax=ax)
    labels = ax.set(xlabel = f"{diagnosis_code[i]}_CAT", ylabel = 'Patient Count', title = f'{diagnosis_code[i]} Diagnosed Patient')
    ax.tick_params(labelrotation=90)

# %% [markdown]
# > From Diagnosis Code 1 to 9, we see that patients distribution are matching. ADMITig diagnosis code is not usually followed through

# %% [markdown]
# ### Selecting one diagnosis code category for analysis and modelling

# %%
# selecting desease as 390-459 or Desease of Circulatory system since it has maximum data
diagnosis_code_cat = [
    col for col in data_inpatient_claims if ("_CAT" in col) & ("DGNS" in col)
]

selected_icd_code = "390-459"
selected_data = data_inpatient_claims.loc[
    data_inpatient_claims[diagnosis_code_cat].isin([selected_icd_code]).any(axis=1), :,
].copy()


# %% [markdown]
# ### Generating target variable

# %%
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

# %% [markdown]
# ### Removing the not readmitted record for the patient which have atleast one admitted record
#

# %%
readmitted_patients_data = selected_data.loc[
    (selected_data["IsReadmitted"] == 1), :
]
readmitted_patients = readmitted_patients_data.loc[:, "DESYNPUF_ID"].unique()

drop_index = selected_data.loc[
    (selected_data["DESYNPUF_ID"].isin(readmitted_patients))
    & (selected_data["IsReadmitted"] == 0)
].index

selected_data.drop(index=drop_index, axis=0, inplace=True)
readmitted_patients_data = selected_data.loc[
    (selected_data["IsReadmitted"] == 1), :
]
not_readmitted_patients_data = selected_data.loc[
    (selected_data["IsReadmitted"] == 0), :
]

# Uncomment the below line to remove duplicate patient from non readmitted data
# not_readmitted_patients_data.drop_duplicates(subset=["DESYNPUF_ID", "IsReadmitted"], keep='last', inplace=True)

final_inpatient_data = pd.concat(
    [readmitted_patients_data, not_readmitted_patients_data], axis=0
)


# %% [markdown]
# ### Dropping raw diagnosis columns

# %%
final_inpatient_data.drop(columns=icd_procedural_code + diagnosis_code, inplace=True)

# %% [markdown]
# ## Processing Outpatient File

# %%
data_outpatient_claim = pd.read_csv("../input/DE1.0 Sample2/DE1_0_2008_to_2010_Outpatient_Claims_Sample_2.zip", parse_dates=['CLM_FROM_DT', 'CLM_THRU_DT'], infer_datetime_format=True)


# %% [markdown]
# ### Checking the columns for NaN and dropping them if NaN are more than a threshold value

# %%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    data_outpatient_claim.isna().sum()[data_outpatient_claim.isna().sum() > 0]/data_outpatient_claim.shape[0]

# %%
data_outpatient_claim.dropna(
    thresh=NON_NAN_THRESHOLD * data_outpatient_claim.shape[0], axis=1, inplace=True
)

# %%
data_outpatient_claim.isna().sum()/data_outpatient_claim.shape[0]

# %% [markdown]
# ### Categorizing PROVIDER NUM

# %%
data_outpatient_claim["PRVDR_NUM"].value_counts()
prvdr_num.get_categories_for_providers(data_outpatient_claim["PRVDR_NUM"])
data_outpatient_claim = data_outpatient_claim.merge(
    right=prvdr_num.unique_prvdr_num_category_df, on=["PRVDR_NUM"], how="left"
)


# %% [markdown]
# ### Creating list of columns for diagnosis, procedure & hcpcs code

# %%
diagnosis_code_out = [
    col for col in data_outpatient_claim.columns if "ICD9_DGNS" in col
]
icd_procedural_code_out = [
    col for col in data_outpatient_claim.columns if "ICD9_PRCDR" in col
]
icd_hcpcs_code_out = [col for col in data_outpatient_claim.columns if "HCPCS_CD" in col]


# %%
data_outpatient_claim[
    diagnosis_code_out + icd_procedural_code_out + icd_hcpcs_code_out
] = (
    data_outpatient_claim[
        diagnosis_code_out + icd_procedural_code_out + icd_hcpcs_code_out
    ]
    .astype("str")
    .replace(["nan", "na"], np.nan)
)


# %% [markdown]
# ### Processing of Procedural Code columns

# %%
proc_code.get_categories_for_procedure_code(
    data_outpatient_claim[icd_procedural_code_out]
)
for col in icd_procedural_code_out:
    data_outpatient_claim[f"{col}"] = data_outpatient_claim[f"{col}"].str[:2]
    data_outpatient_claim[f"{col}_CAT"] = pd.merge(
        left=data_outpatient_claim,
        right=proc_code.unique_procedure_code_category_df,
        left_on=col,
        right_on="Procedure_code",
        how="left",
    )["Procedure_code_CAT"]

# %% [markdown]
# > No columns related to ICD9 procedure codes are remaining after null drops

# %% [markdown]
# ### Processing of HCPCS Code columns

# %%
hcpcs_code = CategorizeCardinalData.HCPCSCodeCategoryCreator()

# %%
replacement_dict = hcpcs_code.__class__.OLD_HCPCS_CODE_MAPPING
data_outpatient_claim.replace(replacement_dict, inplace=True)

# %%
hcpcs_code.get_categories_for_hcpcs_code(
    data_outpatient_claim[icd_hcpcs_code_out]
)
for col in icd_hcpcs_code_out:
    data_outpatient_claim[f"{col}_CAT"] = pd.merge(
        left=data_outpatient_claim,
        right=hcpcs_code.unique_hcpcs_code_category_df,
        left_on=col,
        right_on="HCPCS_code",
        how="left",
    )["HCPCS_code_CAT"]

# %%
fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(20, 20))
plt.subplots_adjust(hspace=1.2)
for i, ax in enumerate(fig.axes):
    subset_data = data_outpatient_claim[["DESYNPUF_ID", f"{icd_hcpcs_code_out[i]}_CAT"]]
    series_plot = data_outpatient_claim.groupby([f"{icd_hcpcs_code_out[i]}_CAT"])["DESYNPUF_ID"].size()
    series_plot = series_plot[series_plot > 100]
    plot = sns.barplot(x=series_plot.index, y=series_plot.values, ax=ax)
    labels = ax.set(xlabel = f"{icd_hcpcs_code_out[i]}_CAT", ylabel = 'Patient Count', title = f'{icd_hcpcs_code_out[i]} Diagnosed Patient')
    ax.tick_params(labelrotation=90)

# %% [markdown]
# ### Categorizing ICD9 Diagnosis Code

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
    )["Diagnosis_code_CAT"]

# %%
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
plt.subplots_adjust(hspace=1.2)
for i, ax in enumerate(fig.axes):
    subset_data = data_outpatient_claim[["DESYNPUF_ID", f"{diagnosis_code_out[i]}_CAT"]]
    series_plot = data_outpatient_claim.groupby([f"{diagnosis_code_out[i]}_CAT"])["DESYNPUF_ID"].size()
#     series_plot = series_plot[series_plot > 100]
    plot = sns.barplot(x=series_plot.index, y=series_plot.values, ax=ax)
    labels = ax.set(xlabel = f"{diagnosis_code_out[i]}_CAT", ylabel = 'Patient Count', title = f'{diagnosis_code_out[i]} Diagnosed Patient')
    ax.tick_params(labelrotation=90)

# %%
data_outpatient_claim.columns = [
    "DESYNPUF_ID" if col == "DESYNPUF_ID" else col + "_OUT"
    for col in data_outpatient_claim
]

# %% [markdown]
# ## Merging outpatient file with inpatient

# %%
final_claim_data = pd.merge(
    left=final_inpatient_data, right=data_outpatient_claim, on="DESYNPUF_ID", how="left"
)



# %% [markdown]
# ### Selecting only those records for which Outpatient claim from date lies between Inpatient Discharge date & Next readmission date

# %%
span_data = final_claim_data.loc[
    (final_claim_data["NCH_BENE_DSCHRG_DT"] <= final_claim_data["CLM_FROM_DT_OUT"])
    & (final_claim_data["CLM_FROM_DT_OUT"] <= final_claim_data["Next_CLM_ADMSN_DT"]), :
]

# %% [markdown]
# ### Create a mask column where selected diagnosis is present

# %%
span_data["ICD9_DGNS_CD_1_MAP_OUT"] = span_data["ICD9_DGNS_CD_1_CAT_OUT"].apply(
    lambda x: "390-459" if x == "390-459" else "Others"
)

span_data["ICD9_DGNS_CD_2_MAP_OUT"] = span_data["ICD9_DGNS_CD_2_CAT_OUT"].apply(
    lambda x: "390-459" if x == "390-459" else "Others"
)

# %% [markdown]
# ### Gettig count of outpatient visit
#

# %%
claim_data = span_data.groupby(["DESYNPUF_ID", "CLM_ID"]).count()
claim_data.reset_index(inplace=True)
claim_data.head()
claim_data = claim_data.loc[:, ["DESYNPUF_ID", "CLM_ID", "SEGMENT"]]

# %% [markdown]
# ### Merge the number of outpatient visit into the inpatient file

# %%
final_inpatient_data = pd.merge(
    left=final_inpatient_data,
    right=claim_data,
    on=["DESYNPUF_ID", "CLM_ID"],
    how="left",
)

final_inpatient_data["SEGMENT_y"].fillna(value=0, inplace=True)

# %%
final_inpatient_data.rename(columns={"SEGMENT_x" : "SEGMENT", "SEGMENT_y" : "NUM_OUTPATIENT_VISIT"}, inplace=True)
final_inpatient_data.head()

# %% [markdown]
# ## Processing of Beneficiary Summary data

# %% [markdown]
# ### Reading Beneficiary File

# %%
# %%bigquery --project dsapac  --use_bqstorage_api data_beneficiary_2008

SELECT * FROM dsapac.CMS.BeneficarySummary_2008

# %%
# %%bigquery --project dsapac  --use_bqstorage_api data_beneficiary_2009

SELECT * FROM dsapac.CMS.BeneficarySummary_2009

# %%
# %%bigquery --project dsapac  --use_bqstorage_api data_beneficiary_2010

SELECT * FROM dsapac.CMS.BeneficarySummary_2010

# %%
data_beneficiary_2008["YEAR"] = pd.to_datetime("2008-12-31", infer_datetime_format=True)
data_beneficiary_2009["YEAR"] = pd.to_datetime("2009-12-31", infer_datetime_format=True)
data_beneficiary_2010["YEAR"] = pd.to_datetime("2010-12-31", infer_datetime_format=True)

# %% [markdown]
# ### Combining all the beneficiary file

# %%
combined_beneficiary_data_2 = pd.concat(
    [data_beneficiary_2008, data_beneficiary_2009, data_beneficiary_2010,], axis=0,
)



# %% [markdown]
# ### Correcting dtypes

# %%
combined_beneficiary_data_2['BENE_BIRTH_DT'] =  pd.to_datetime(combined_beneficiary_data_2['BENE_BIRTH_DT'], format="%Y%m%d")
combined_beneficiary_data_2['BENE_DEATH_DT'] =  pd.to_datetime(combined_beneficiary_data_2['BENE_DEATH_DT'], format="%Y%m%d")

# %% [markdown]
# ### Calculating Beneficiary Age

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


# %% [markdown]
# ### Creating year column for merging with inpatient file

# %%
combined_beneficiary_data_2["YEAR"] = combined_beneficiary_data_2["YEAR"].dt.year

# %% [markdown]
# ### Combining state and county to a single column

# %%
combined_beneficiary_data_2["BENE_STATE_COUNTY_CODE"] = (
    combined_beneficiary_data_2["SP_STATE_CODE"].astype(str)
    + "-"
    + combined_beneficiary_data_2["BENE_COUNTY_CD"].astype(str)
)

combined_beneficiary_data_2.drop(
    columns=["SP_STATE_CODE", "BENE_COUNTY_CD"], inplace=True, axis=1
)

# %% [markdown]
# ### Correcting datatypes

# %%
categorical_columns = ['BENE_SEX_IDENT_CD', 'BENE_RACE_CD', 'BENE_ESRD_IND', 'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA', 'BENE_STATE_COUNTY_CODE', 'YEAR']


# %%
combined_beneficiary_data_2[categorical_columns] = combined_beneficiary_data_2[categorical_columns].astype("category")

# %%
numerical_columns = combined_beneficiary_data_2.select_dtypes(exclude=['category', 'object', 'datetime']).columns

# %% [markdown]
# ## Deleting unnecessary objects / references

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
gc.collect()

# %% [markdown]
# ## Creating Final Dataframe

# %%
final_inpatient_data.columns = [
    "DESYNPUF_ID" if col == "DESYNPUF_ID" else col + "_INP"
    for col in final_inpatient_data
]



# %%
#Merging final inpatient & beneficiary data to create 
final_df = pd.merge(
    left=combined_beneficiary_data_2,
    right=final_inpatient_data,
    left_on=["DESYNPUF_ID", "YEAR"],
    right_on=["DESYNPUF_ID", "CLAIM_YEAR_INP"],
    how="inner",
)


# %% [markdown]
# # Final Data Preparation

# %% [markdown]
# ## Bigquery Processing

# %% [markdown]
# Following are the steps that have been performed using bigquery and the results stored as a bigquery table
# - Selected records for required disease category (i.e. Disease of the Circulatory System).
# - Generated target column in the inpatient records using discharge date and the next admission date.
# - Merged the above records with outpatient table to get count of outpatient data visit between discharge date and next admission date
# - Merged the above records with beneficiary summary data to get the final dataframe

# %%
# %%bigquery --project dsapac  --use_bqstorage_api final_df

SELECT * FROM dsapac.CMS.InpatientTarget

# %% [markdown]
# ### Dropping NaN records

# %%
final_df.dropna(thresh=NON_NAN_THRESHOLD*final_df.shape[0], axis=1, inplace=True)
print("Final missing data count per columns")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    final_df.isna().sum()[final_df.isna().sum() > 0]/final_df.shape[0]

# %% [markdown]
# ### Categorizing Provider Number, Diagnosis Codes and Procedure code

# %% [markdown]
# #### Processing of PRVDR_NUM column.

# %%
prvdr_num = CategorizeCardinalData.ProviderNumCategoryCreator()
prvdr_num.get_categories_for_providers(final_df["PRVDR_NUM_INP"])
final_df = final_df.merge(
    right=prvdr_num.unique_prvdr_num_category_df, left_on=["PRVDR_NUM_INP"], right_on=["PRVDR_NUM"], how="left"
)

# %% [markdown]
# #### Preprocessing for Procedure and Diagnosis Code

# %%
diagnosis_code = [col for col in final_df.columns if "ICD9_DGNS" in col]
icd_procedural_code = [
    col for col in final_df.columns if "ICD9_PRCDR" in col
]
icd_hcpcs_code = [col for col in final_df.columns if "HCPCS_CD" in col]

# %%
# Preparing dataframe for categorizing
final_df[diagnosis_code + icd_procedural_code + icd_hcpcs_code] = (
    final_df[diagnosis_code + icd_procedural_code + icd_hcpcs_code]
    .astype("str")
    .replace(["nan", "na"], np.nan)
)

# %% [markdown]
# #### Processing of Procedure Code columns.

# %%
# Processing of Procdural Code columns
proc_code = CategorizeCardinalData.ProcedureCodeCategoryCreator()
proc_code.get_categories_for_procedure_code(final_df[icd_procedural_code])
for col in icd_procedural_code:
    final_df[f"{col}"] = final_df[f"{col}"].str[:2]
    final_df[f"{col}_CAT"] = pd.merge(
        left=final_df,
        right=proc_code.unique_procedure_code_category_df,
        left_on=col,
        right_on="Procedure_code",
        how="left",
    )["Procedure_code_CAT"]

# %% [markdown]
# #### Processing of Diagnosis Code column.

# %%
# Processing of Diagnosis Code
diag_code = CategorizeCardinalData.DiagnosisCodeCategoryCreator()
diag_code.get_categories_for_diagnosis_code(final_df[diagnosis_code])
for col in diagnosis_code:
    final_df[f"{col}"] = final_df[f"{col}"].str[:3]
    final_df[f"{col}_CAT"] = pd.merge(
        left=final_df,
        right=diag_code.unique_diagnosis_code_category_df,
        left_on=col,
        right_on="Diagnosis_code",
        how="left",
    )["Diagnosis_code_CAT"]

# %%
final_df.drop(columns=diagnosis_code + icd_procedural_code + icd_hcpcs_code, inplace=True)

# %%
category_column = [col for col in final_df if "CAT" in col]

# %% [markdown]
# ### Distribution of Target in final_data

# %%
final_df['IsReadmitted'].value_counts()/final_df.shape[0]

# %% [markdown]
# ## Creating list of categorical data

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
    #"BENE_STATE_COUNTY_CODE",
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
] + category_column



# %%
final_df.columns

# %% [markdown]
# ## List of columns to drop

# %%
# PRVDR_NUM_INP : Category column exist PRVDR_NUM_CAT_INP
# CLM_DRG_CD_INP : Claim Diagnosis Related Group Code not relevant for Readmission detection
# PRVDR_NUM_OUT : Category column exists PRVDR_NUM_CAT_OUT
# 'ICD9_DGNS_CD_1_OUT', 'ICD9_DGNS_CD_2_OUT', 'HCPCS_CD_1_OUT', 'HCPCS_CD_2_OUT', 'HCPCS_CD_3_OUT' : Category column exists
# 'HCPCS_CD_1_CAT_DESC_OUT', 'HCPCS_CD_2_CAT_DESC_OUT', 'HCPCS_CD_3_CAT_DESC_OUT' : Description column to be used later

cols_to_drop = ['YearofAdmission', 'TimeInHospital', 'RelativeDiseaseCount', 'CLM_ID_INP', ]
date_cols = list(final_df.select_dtypes(include="datetime").columns)
npi_cols = [col for col in final_df.select_dtypes(include="number") if "NPI" in col]



# %%
df = final_df.copy()


# %% [markdown]
# ## Converting required columns to category

# %%
df[categorical_features] = df[categorical_features].astype("category")



# %% [markdown]
# ## Dropping unnecessary columns

# %%
df.drop(columns=cols_to_drop + date_cols + npi_cols, inplace=True, axis=1)


# %%
# Check - 1 dropping columns as per feature importance
# df.drop(columns="BENRES_IP", inplace=True, axis=1)


# %% [markdown]
# ## Feature and Target data separation

# %%
X = df.drop(columns=["IsReadmitted"], axis=1)
y = df.loc[:, "IsReadmitted"]



# %% [markdown]
# ## Creating train and test set

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE_FOR_SPLIT, random_state=RANDOM_STATE
)



# %%
categorical_features = list(X.select_dtypes(include="category").columns)
numerical_features = list(X.select_dtypes(include="number").columns)



# %% [markdown]
# ## Checking for Correlation between dataset

# %% [markdown]
# ### Pearson Correlation on Numerical features 

# %%
plt.figure(figsize=(15, 10))
sns.heatmap(X_train.corr(), annot=True, fmt=".2f", cmap='magma')

# %%
# dropping highly correlated features
check_1 = X_train.drop(columns="MEDREIMB_OP")

# %% jupyter={"outputs_hidden": true}
plt.figure(figsize=(15, 10))
sns.heatmap(check_1.corr(), annot=True, fmt=".2f", cmap='magma')

# %%
# Dropping another correlated pair
check_2 = check_1.drop(columns="MEDREIMB_CAR")

# %% jupyter={"outputs_hidden": true}
plt.figure(figsize=(15, 10))
sns.heatmap(check_2.corr(), annot=True, fmt=".2f", cmap='magma')

# %%
# dropping another highly correlated feature
check_3 = check_2.drop(columns="NCH_PRMRY_PYR_CLM_PD_AMT_INP")

# %%
plt.figure(figsize=(15, 10))
sns.heatmap(check_3.corr(), annot=True, fmt=".2f", cmap='magma')

# %%
del  final_df, X, y, check_1, check_2, check_3
gc.collect()

# %%
# # Very high correlation between ('BENRES_OP', 'MEDREIMB_OP'), ('NCH_PRMRY_PYR_CLM_PD_AMT_INP', PPPYMT_IP) & ('BENRES_CAR', 'MEDREIMB_CAR') hence adding one of them to correlated list for dropping one column in the pair

correlated_cols_drop_list = [
    "MEDREIMB_OP",
    "MEDREIMB_CAR",
    "NCH_PRMRY_PYR_CLM_PD_AMT_INP"
]


# %% [markdown]
# ### Running Chi2Test on Categorical Data

# %%
ct = StatisticalTest.ChiSquare(pd.concat([X_train, y_train], axis=1))


# %% [markdown]
# ### Checking for importance w.r.t Target

# %%
for c in categorical_features:
    ct.TestIndependence(c, "IsReadmitted")


# %%
cramers = pd.DataFrame(
    {
        i: [ct.cramers_v(i, j) for j in (categorical_features + ['IsReadmitted'])]
        for i in categorical_features
    }
)
cramers["column"] = [i for i in (categorical_features + ['IsReadmitted']) if i not in ["memberid"]]
cramers.set_index("column", inplace=True)

# %%
# categorical correlation heatmap
plt.figure(figsize=(25, 25))
sns.heatmap(cramers, annot=True, fmt=".2f", cmap="magma")
plt.show()

# %%
# # > High correlation between BENE_STATE_COUNTY_CODE & PRVDR_NUM_CAT_INP columns

# # %%
# # X_train.drop(columns=["BENE_STATE_COUNTY_CODE"], axis=1, inplace=True)
# # X_test.drop(columns=["BENE_STATE_COUNTY_CODE"], axis=1, inplace=True)


# %% [markdown]
# ## Final features drop list 

# %%
dropped_features = correlated_cols_drop_list + ['SP_STATE_CODE', 'BENE_COUNTY_CD']
dropped_features

# %% [markdown]
# ## Section4–Predictive Modelling&Evaluation
#
# ### L1. Clearly define features& Outcomefor modelling for hospital readmissionsand train basic model.

# %%
print("Features in the model are : " + X_train.columns)
print("Outcome of the model is " + y_train.name)

# %% [markdown]
# ### L2. Train binary classification model, mention contributing features and their importance for predictions

# %%
df_stats = df.copy()

# %%
map_comorbidities = {1 : "Yes", 2 : 'No'}
map_target = {1 : "Readmitted", 0: "Not Readmitted"}

# %%
for col in [col for col in df_stats if "SP" in col]:
    df_stats[col] = df_stats[col].map(map_comorbidities)

# %%
df_stats['IsReadmitted'] = df_stats['IsReadmitted'].map(map_target)

# %%
s = ""
for col in df_stats.select_dtypes(include='number'):
    print(f'{col}', end=" + ")
for col in df_stats.select_dtypes(include='category'):
    print(f'C({col})', end=" + ")

# %%
formula = """IsReadmitted ~  BENE_HI_CVRAGE_TOT_MONS + BENE_SMI_CVRAGE_TOT_MONS + BENE_HMO_CVRAGE_TOT_MONS + 
PLAN_CVRG_MOS_NUM + MEDREIMB_IP + BENRES_IP + PPPYMT_IP + MEDREIMB_OP + BENRES_OP + PPPYMT_OP + MEDREIMB_CAR + BENRES_CAR + PPPYMT_CAR +
Age + CLM_PMT_AMT_INP + NCH_PRMRY_PYR_CLM_PD_AMT_INP + CLM_PASS_THRU_PER_DIEM_AMT_INP + NCH_BENE_IP_DDCTBL_AMT_INP + NCH_BENE_PTA_COINSRNC_LBLTY_AM_INP + NCH_BENE_BLOOD_DDCTBL_LBLTY_AM_INP + 
CLM_UTLZTN_DAY_CNT_INP + TotalOutpatientVist + 
C(BENE_RACE_CD) + C(BENE_SEX_IDENT_CD) + C(BENE_ESRD_IND) + 
C(SP_ALZHDMTA) + C(SP_CHF) + C(SP_CHRNKIDN) + C(SP_CNCR) + C(SP_COPD) + C(SP_DEPRESSN) + C(SP_DIABETES) + C(SP_ISCHMCHT) + C(SP_OSTEOPRS) + C(SP_RA_OA) + C(SP_STRKETIA)
+ C(PRVDR_NUM_CAT) + 
C(ICD9_PRCDR_CD_1_INP_CAT) + 
C(ICD9_DGNS_CD_1_INP_CAT) """

# %%
model = sm.GLM.from_formula(formula, family=sm.families.Binomial(), data=df_stats)

# %% jupyter={"outputs_hidden": true}
result = model.fit()

# %%
result.summary()

# %% [markdown]
# ## Modelling

# %% [markdown]
# ### Initializing objects for Preprocessing, Oversampling and Modelling

# %%
# Initializing all the objects for preprocessing
std_scalar = StandardScaler()
min_max_scalar = MinMaxScaler()
onehot_encoder = OneHotEncoder(drop="first", sparse=False)
median_imputer = SimpleImputer(strategy="median", missing_values=np.nan)
constant_imputer = CategoricalVariableImputer(fill_value=MISSING_VALUE_LABEL)
ordinal_encoder = OrdinalEncoder()



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


# %%
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


# %% [markdown]
# ### Wrapper class for modelling, prediction and metrics

# %%
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
                "True_Negative",
                "False_Positive",
                "False_Negative",
                "True_Positive"
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
                "True_Negative",
                "False_Positive",
                "False_Negative",
                "True_Positive"
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
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()
        self.current_metrics["True_Negative"] = tn
        self.current_metrics["False_Positive"] = fp
        self.current_metrics["False_Negative"] = fn
        self.current_metrics["True_Positive"] = tp
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


# %% [markdown]
# ### Creating Preprocessing pipelines

# %% [markdown]
# ### Creating temp files for storing cache

# %%
# Temp files for caching Pipelines
numerical_cachedir = mkdtemp(prefix="num")
numerical_memory = Memory(location=numerical_cachedir, verbose=VERBOSE_PARAM_VALUE)

categorical_cachedir = mkdtemp(prefix="cat")
categorical_memory = Memory(location=categorical_cachedir, verbose=VERBOSE_PARAM_VALUE)

pipeline_cachedir = mkdtemp(prefix="cat")
pipeline_memory = Memory(location=pipeline_cachedir, verbose=VERBOSE_PARAM_VALUE)



# %% [markdown]
# ### Pipeline for transforming numerical data

# %%
set_config(display='diagram')

# %%

# Pipeline to automate the numerical column processing
numerical_transformer = Pipeline(
    steps=[("imputer_with_medium", median_imputer), ("scaler", std_scalar)],
    verbose=VERBOSE_PARAM_VALUE,
    memory=numerical_memory,
)

numerical_transformer

# %% [markdown]
# ### Pipelines for transforming categorical data

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

categorical_transformer

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

ord_categorical_transformer

# %% [markdown]
# ### Transformer for running numerical and categorical pipeline on appropriate columns

# %%
# Transformer to run both the numerical and one hot encoder pipeline on specified dtypes

# preprocessing_transformer = ColumnTransformer(
#     [
#         (
#             "categorical",
#             categorical_transformer,
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

# Transformer to run both the numerical and ordinal encoder pipeline on specified dtypes
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

preprocessing_transformer

# %% [markdown]
# ### Final Preprocessing Pipeline without class balancing

# %% jupyter={"source_hidden": true}

# Pipeline to automate drop of specified column and running the preprocessor for the remaining column
preprocessing_pipeline = Pipeline(
    [
        (
            "column selector",
            SelectColumnsTransfomer(columns=dropped_features, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)


preprocessing_pipeline

# %% [markdown]
# ### Final Preprocessing Pipeline with class balancing

# %% jupyter={"source_hidden": true}
imblanced_preprocessing_pipeline_smote = imb_pipeline.Pipeline(
    [
        (
            "column selector",
            SelectColumnsTransfomer(columns=dropped_features, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE oversampling", smt),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

imblanced_preprocessing_pipeline_randover = imb_pipeline.Pipeline(
    [
        (
            "column selector",
            SelectColumnsTransfomer(columns=dropped_features, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random oversampling", random_oversampling),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

imblanced_preprocessing_pipeline_smote
imblanced_preprocessing_pipeline_randover

# %% [markdown]
# ### Runnig pipelines on train and test set

# %%
X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
X_train_transformed = pd.DataFrame(
    X_train_transformed, columns=get_ct_feature_names(preprocessing_transformer)
)


# %%
X_test_transformed = preprocessing_pipeline.transform(X_test)
X_test_transformed = pd.DataFrame(
    X_test_transformed, columns=get_ct_feature_names(preprocessing_transformer)
)

# %% jupyter={"outputs_hidden": true}
(
    X_train_transformed_smote,
    y_train_transformed_smote,
) = imblanced_preprocessing_pipeline_smote.fit_resample(X_train, y_train)


# %%
X_train_transformed_smote = pd.DataFrame(
    X_train_transformed_smote, columns=get_ct_feature_names(preprocessing_transformer),
)


# %% jupyter={"outputs_hidden": true}
(
    X_train_transformed_randover,
    y_train_transformed_randover,
) = imblanced_preprocessing_pipeline_randover.fit_resample(X_train, y_train)


# %%
X_train_transformed_randover = pd.DataFrame(
    X_train_transformed_randover,
    columns=get_ct_feature_names(preprocessing_transformer),
)

# %%
y_train.value_counts(),y_train_transformed_smote.value_counts(), y_train_transformed_randover.value_counts()

# %% [markdown]
# ### Creating wrapper class objects for training pipelines

# %%
pipeline_trainer = ModelTrainer(
    xtrain=X_train, ytrain=y_train, xtest=X_test, ytest=y_test
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


# %% jupyter={"source_hidden": true}
ct = StatisticalTest.ChiSquare(pd.concat([X_train_transformed_smote, y_train_transformed_smote], axis=1))
for c in categorical_features:
    ct.TestIndependence(c, "IsReadmitted")

# %% [markdown]
# ### Creating pipelines with estimators

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
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
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
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
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
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE oversampling", smt),
        ("LogReg_Classifier", lr),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

logreg_randover_pipeline = imb_pipeline.Pipeline(
    [
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random oversampling", random_oversampling),
        ("LogReg_Classifier", lr),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)



# %%
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
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
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
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random oversampling", random_oversampling),
        ("Feature Selection", rfe_dt),
        ("DT_Classifier", dt),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

# %%
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
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
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
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random oversampling", random_oversampling),
        ("Feature Selection", rfe_rfc),
        ("RFC_Classifier", rfc),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)


# %%
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
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
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
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
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
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE Sampling", smt),
        ("Classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

dt_randover_pipeline = imb_pipeline.Pipeline(
    [
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random OverSampling", random_oversampling),
        ("Classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

# %%
rfc_pipeline = Pipeline(
    [("Preprocessing Step", preprocessing_pipeline), ("rfc_classifier", rfc),],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

rfc_smote_pipeline = imb_pipeline.Pipeline(
    [
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE Sampling", smt),
        ("rfc_classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

rfc_randover_pipeline = imb_pipeline.Pipeline(
    [
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random OverSampling", random_oversampling),
        ("rfc_classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

# %%
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
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
        ("SMOTE Sampling", smt),
        ("extratrees_classifier", extratrees_clf),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)

extratrees_randover_pipeline = imb_pipeline.Pipeline(
    [
        (
            "drop columns",
            SelectColumnsTransfomer(columns=correlated_cols_drop_list, drop=True),
        ),
        ("Preprocessing Step", preprocessing_transformer),
        ("Random OverSampling", random_oversampling),
        ("extratrees_classifier", extratrees_clf),
    ],
    verbose=VERBOSE_PARAM_VALUE,
    memory=pipeline_memory,
)


# %% [markdown]
# ### Running the model pipelines and validating results

# %%
# pipeline_trainer.train_test_model(logreg_rfe_smote_pipeline)

# #%%
# pipeline_trainer.train_test_model(logreg_rfe_pipeline)
# #%%
# pipeline_trainer.train_test_model(logreg_rfe_randover_pipeline)


# %%
dt_rfe_smote_pipeline
pipeline_trainer.train_test_model(dt_rfe_smote_pipeline)


# %% jupyter={"outputs_hidden": true}
dt_rfe_pipeline
pipeline_trainer.train_test_model(dt_rfe_pipeline)

# %% jupyter={"outputs_hidden": true}
dt_rfe_randover_pipeline
pipeline_trainer.train_test_model(dt_rfe_randover_pipeline)


# %%
pipeline_trainer.train_test_model(rfc_rfe_smote_pipeline)


# %%
rfc_rfe_pipeline
pipeline_trainer.train_test_model(rfc_rfe_pipeline)

# %%
pipeline_trainer.optimize_threshold()

# %% jupyter={"outputs_hidden": true}
rfc_rfe_randover_pipeline
pipeline_trainer.train_test_model(rfc_rfe_randover_pipeline)

# %%
pipeline_trainer.train_test_model(logreg_smote_pipeline)


# %%
pipeline_trainer.train_test_model(logreg_pipeline)

# %%
pipeline_trainer.train_test_model(logreg_randover_pipeline)


# %%
pipeline_trainer.train_test_model(dt_smote_pipeline)


# %% jupyter={"outputs_hidden": true}
dt_pipeline
pipeline_trainer.train_test_model(dt_pipeline)

# %% jupyter={"outputs_hidden": true}
dt_randover_pipeline
pipeline_trainer.train_test_model(dt_randover_pipeline)


# %%
pipeline_trainer.train_test_model(rfc_smote_pipeline)


# %%
rfc_pipeline
pipeline_trainer.train_test_model(rfc_pipeline)

# %%
pipeline_trainer.optimize_threshold()

# %% jupyter={"outputs_hidden": true}
rfc_randover_pipeline
pipeline_trainer.train_test_model(rfc_randover_pipeline)


# %%
pipeline_trainer.train_test_model(extratrees_smote_pipeline)


# %% jupyter={"outputs_hidden": true}
extratrees_pipeline
pipeline_trainer.train_test_model(extratrees_pipeline)

# %% jupyter={"outputs_hidden": true}
extratrees_randover_pipeline
pipeline_trainer.train_test_model(extratrees_randover_pipeline)

# %%
pipeline_trainer.train_test_model(extratrees_rfe_smote_pipeline)


# %%
extratrees_rfe_pipeline
pipeline_trainer.train_test_model(extratrees_rfe_pipeline)

# %%
pipeline_trainer.optimize_threshold()

# %% jupyter={"outputs_hidden": true}
extratrees_rfe_randover_pipeline
pipeline_trainer.train_test_model(extratrees_rfe_randover_pipeline)


# %%
dump(
    rfc_pipeline, r"../models/rfc_pipeline_ordinal.pkl"
)

# %% jupyter={"outputs_hidden": true}
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1):
    pipeline_trainer.metric_df

# %%
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

# %% [markdown]
# # L3. Please articulate the following business questions based on findings

# %% [markdown]
# ## Where do providers focus on improving the efficacy of their system?

# %% [markdown]
# 1. Proper initial diagnosis - In many of the cases ADMTING_ICD9_DIAGNOSIS_CODE differs from the actual diagnosis
# 2. Patient followup post discharge. Patient having high no. of outpatient visit (follow-ups) have less chances of readmission
# 3. Special attention to old patient and patients with co-morbidity
# 4. Medication counseling by pharmacists
# 5. Self-management education programs
# 6. Tele-homecare and remote patient monitoring
# 7. Specialized discharge teams and discharge planners

# %% [markdown]
# ## What's cost saving and projection for next few years if providers utilize these models?

# %% [markdown]
# 1. Aggressive approach to prevent readmission, model will more likely identify more than half of the patients that would be readmitted
# 2. A study conducted in Baylor Medical Center Garland (BMCG), under the current payment system, i.e., fee-for-service, the intervention reduced the hospital financial contribution margin, i.e., revenues minus costs, on average $227 for each Medicare patient with HF 

# %% [markdown]
# ## In management of healthcare institutions like hospitals, should they focus on efficacy? Or should we focus on effectiveness of services?

# %% [markdown]
# 1. In effectiveness we focus on outcomes (outcomes such as patient recovery, patient satisfaction), while in efficacy, we focus in consuming less resources (such as capital and human resources).
# 2. The Health Care Institutions need to focus in both. Without efficacy, there would never be effectiveness. Efficacy must be put in place first. This if properly implemented would guarantee effectiveness. Well implemented efficacy gives rise to effectiveness.

# %% [markdown]
# # L4. Train any binary classification model by following these steps 

# %% [markdown]
# ## Apply Any dimensionality reduction technique (PCA ,tsne,UMAP)

# %% [markdown]
#  > Refer the clustering section - PCA used to reduction dimensionality to 2 for plotting

# %% [markdown]
# ## Apply any unsupervised clustering technique

# %% [markdown]
#  Refer the clustering section
#  - KMeans clustering used
#  - 15 clusters observed using elbow method
#  - plot of the cluster

# %% [markdown]
# ## Train binary classification model

# %% [markdown]
# Refer the Modelling section
# - Pipelines created for preprocessing
# - Separate pipelines to supporting both OneHotEncoder and Ordinal Encoder
# - Separate pipelines to provide class balancing using SMOTE and Randoversampling
# - Decision Tree, RandomForest & ExtraTrees classifier used

# %% [markdown]
# ## Tune the hyperparameters as required for algorithms

# %% [markdown]
# Refer above

# %% [markdown]
# ## Identify the best threshold and reasoning

# %% [markdown]
# Best threshold is 0.35 for ExtraTrees pipeline, 0.341 for RandomForest pipeline with and without RFE for max 30 features

# %% [markdown]
# ### Report model performance on following KPIs (AUC, AUCPR, Sensitivity, Specificity, PPV,NPV) and how provider can take decision based on these metrics?

# %% [markdown]
# RandomForest Pipeline

# %%
pipeline_trainer.metric_df.iloc[0, :]

# %%
ppv = pipeline_trainer.metric_df.loc[0, "True_Positive"]/(pipeline_trainer.metric_df.loc[0, "True_Positive"] + pipeline_trainer.metric_df.loc[0, "False_Positive"])

# %%
npv = pipeline_trainer.metric_df.loc[0, "True_Negative"]/(pipeline_trainer.metric_df.loc[0, "True_Negative"] + pipeline_trainer.metric_df.loc[0, "False_Negative"])

# %%
print(f"Model's PPV & NPV are {ppv} & {npv} respectively")

# %% [markdown]
# ### 5. Can you run the model for next 5 years based on the data you have used to train? What all implications and how would provider be assured model is good enough to put it in real world ?

# %% [markdown]
# ### 6. Comment on model drift 

# %% [markdown]
# No. Model output needs to be continously monitored along with the data and its statistical properties.
# While training the model, we assume that the parameters learned by the model during training period would remain constant w.r.t time. However, we know that to not be the case. The trend of population as well as its parameters changes over time making our model output invalid.
# Eg. The defination of readmission may change in next year and the model would fail and would need to be retrained.
#
# Model drift : What it essentially means is that the relationship between the target variable and the independent variables changes with time. Due to this drift, the model keeps becoming unstable and the predictions keep on becoming erroneous with time.
# Model drift can be classified into two broad categories. The first type is called ‘concept drift’. This happens when the statistical properties of the target variable itself change.
# The second and the more common type is ‘data drift’. This happens when the statistical properties of the predictors change. Again, if the underlying variables are changing due seasonality, etc., the model is bound to fail.

# %% [markdown]
# # Clustering

# %% [markdown]
# ## Initializing objects for unsupervised learning

# %%
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

# %% [markdown]
# ## Finding number of clusters using elbow method

# %% jupyter={"outputs_hidden": true}
wcss = []

for i in range(1, 25):
    km = KMeans(
        n_clusters=i,
        init="k-means++",
        max_iter=500,
        n_init=20,
        random_state=0,
        verbose=VERBOSE_PARAM_VALUE,
    )
    print(f"Fitting started for {i}")
    km.fit(X_train_transformed)
    print(f"Fitting completed for {i}")
    wcss.append(km.inertia_)

# %%

plt.figure(figsize=(20, 20))
sns.lineplot(x=range(1, 25), y=wcss, marker="X", linewidth=2, markersize=12)
plt.show()

# %% [markdown]
# ## Running clustering with selected no. of clusters

# %%
# Initialize clustering objects
kmeans_clustering = KMeans(
    n_clusters=15,
    verbose=VERBOSE_PARAM_VALUE,
    init="k-means++",
    max_iter=300,
    random_state=RANDOM_STATE,
    n_init=10,
)

mini_batch_kmeans = MiniBatchKMeans(
    n_clusters=15,
    verbose=VERBOSE_PARAM_VALUE,
    random_state=RANDOM_STATE,
    batch_size=20000,
    max_iter=10,
)

agglomeratrive_clustering = AgglomerativeClustering()

# %%
X_train_transformed_clustering = X_train_transformed.copy()

# %%
kmeans_clustering.fit(X_train_transformed_clustering)

# %%
X_train_transformed_clustering['clusters'] = kmeans_clustering.predict(X_train_transformed_clustering)

# %%
X_train_transformed_clustering['clusters'].value_counts()

# %%
pca = PCA(n_components=2, random_state=RANDOM_STATE)

# %%
d2_visual = pca.fit_transform(X_train_transformed_clustering.iloc[ :, :-1])

# %%
d2_visual[ : , 0]
d2_visual[ : , 1]


# %%
df = pd.DataFrame(d2_visual, columns=['pc1', 'pc2'])

# %%
df['clusters'] = X_train_transformed_clustering['clusters']

# %%
centers = kmeans_clustering.cluster_centers_

# %%
N = 36632
x, y = d2_visual[ : , 0], d2_visual[ : , 1]
s = np.random.randint(10, 220, size=N)

fig, ax = plt.subplots()

scatter = ax.scatter(x, y, c=X_train_transformed_clustering['clusters'])
scatter = ax.scatter(kmeans_clustering.cluster_centers_[:, 0], kmeans_clustering.cluster_centers_[:, 1], s=300, c='red', label = 'Centroids')

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="Cluster")
ax.add_artist(legend1)

# produce a legend with a cross section of sizes from the scatter
# handles, labels = scatter.legend_elements(prop="", alpha=0.6)
legend2 = ax.legend(handles, labels, loc="upper left", title="Sizes")

plt.show()

# %%
N = 36632
x, y = d2_visual[ : , 0], d2_visual[ : , 1]
s = np.random.randint(10, 220, size=N)

fig, ax = plt.subplots()

scatter = ax.scatter(x, y, c=X_train_transformed_clustering['clusters'])
scatter = ax.scatter(kmeans_clustering.cluster_centers_[:, 0], kmeans_clustering.cluster_centers_[:, 1], s=300, c='red', label = 'Centroids')

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="Cluster")
ax.add_artist(legend1)

# produce a legend with a cross section of sizes from the scatter
# handles, labels = scatter.legend_elements(prop="", alpha=0.6)
legend2 = ax.legend(handles, labels, loc="upper left", title="Sizes")

plt.show()

# %% [markdown]
# # Section 5 : Model Opertionalization 

# %% [markdown]
# ## L3.Use any trained model/create new model , save it as pickle fileformat and expose end points as rest api.

# %% [markdown]
# Run the app.py file
