#%%
import pandas as pd
import numpy as np

#%%
import os

cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)


# %%
import sqlite3
from GetICDCode import ICD9Codes

#%%
conn_object = sqlite3.connect("..\database\cms_data.db")
fetch_diagnosis = ICD9Codes()
#%%
data_inpatient_claims_2 = pd.read_csv(
    "..\input\DE1.0 Sample2\DE1_0_2008_to_2010_Inpatient_Claims_Sample_2.zip",
)
# %%
icd_procedural_features = [
    col for col in data_inpatient_claims_2.columns if "ICD9_PRCDR" in col
]
# %%
procedural_code = pd.DataFrame()
for col in icd_procedural_features:
    procedural_code = pd.concat(
        [procedural_code, pd.DataFrame(data_inpatient_claims_2[col].unique())], axis=0
    )


# %%
unique_procedural_code = pd.DataFrame(
    procedural_code[0].unique(), columns=["Procedural_code"]
)


# %%
unique_procedural_code.dropna(axis=0, inplace=True)
# %%
unique_procedural_code["description"] = unique_procedural_code["Procedural_code"].apply(
    lambda x: fetch_diagnosis.get_description_for_icd_code(
        codeType="icd9pcs", ICD9Codes=x
    )
)

#%%
unique_procedural_code.to_sql(
    "ICD9_Procedural_Code_Mapping", con=conn_object, if_exists="append"
)
