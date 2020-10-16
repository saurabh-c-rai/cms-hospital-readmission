# %%
import pandas as pd
import numpy as np

#%%
import json

# %%
# Custom packages
from utils import CustomPipeline, ExtractData, feature_importance, StatisticalTest

#%%
# loading configuration values
with open("../config.json", "r") as f:
    config = json.load(f)

#%%
NON_NAN_THRESHOLD = config[0]["threshold_nan"]


# %%
data = ExtractData.FetchSubset(no_of_subset=1)

# %%
data_inpatient_claims = data.fetchFromInpatientDataset()
# %%
data_inpatient_claims.isna().sum() / data_inpatient_claims.shape[0]
# %%
data_inpatient_claims.dropna(thresh=NON_NAN_THRESHOLD, axis=1, inplace=True)
# %%
