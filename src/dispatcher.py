#%%
from imblearn import pipeline as imb_pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
import numpy as np
from sklearn.compose import ColumnTransformer

#%%
import os

cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)
#%%
from tempfile import mkdtemp
from matplotlib import pyplot as plt
import seaborn as sns

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

#%%
from joblib import Memory, dump, load

#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier

#%%
from sklearn.compose import ColumnTransformer, make_column_selector

#%%
import json

#%%
from utils.CustomPipeline import CardinalityReducer, get_ct_feature_names

#%%
with open("../config.json", "r") as f:
    config = json.load(f)

#%%
# constant variables
INFREQUENT_CATEGORY_CUT_OFF = config[3]["INFREQUENT_CATEGORY_CUT_OFF"]
INFREQUENT_CATEGORY_LABEL = config[3]["INFREQUENT_CATEGORY_LABEL"]
RANDOM_STATE = config[3]["RANDOM_STATE"]
MISSING_VALUE_LABEL = config[3]["MISSING_VALUE_LABEL"]
EDA_REPORT_LOCATION = config[1]["eda_report_location"]
MODEL_REPOSITORY_LOCATION = config[1]["model_repository_location"]
#%%
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
        (
            "infrequent_category_remover",
            CardinalityReducer(
                cutt_off=INFREQUENT_CATEGORY_CUT_OFF, label=INFREQUENT_CATEGORY_LABEL
            ),
        ),
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
# Model initialization
lr = LogisticRegression(random_state=RANDOM_STATE)
lda = LinearDiscriminantAnalysis()
dt = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
rfc = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
#%%
# %%
# Oversampler initialization
smt = SMOTE(random_state=RANDOM_STATE, n_jobs=-1)
random_oversampling = RandomOverSampler(random_state=RANDOM_STATE)

#%%
# Feature selector initialization
rfe_lr = RFE(estimator=lr, n_features_to_select=20, verbose=1, step=0.1)
rfe_dt = RFE(estimator=dt, n_features_to_select=20, verbose=1, step=0.1)
rfe_rfc = RFE(estimator=rfc, n_features_to_select=20, verbose=1, step=0.1)

#%%


def getMetricsData(y_true: np.array, y_pred: np.array):
    print(f"Accuracy score is {accuracy_score(y_true, y_pred)}")
    print(f"Precision score is {precision_score(y_true, y_pred)}")
    print(f"Recall score is {recall_score(y_true, y_pred)}")
    print(f"F1 score is {f1_score(y_true, y_pred)}")
    print(f"AUC ROC score is {roc_auc_score(y_true, y_pred)}")
    print(f"Classification Report is \n {classification_report(y_true, y_pred)}")
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap="Blues", fmt="d")
    plt.show()


#%%


preprocessing_pipeline = ColumnTransformer(
    [
        (
            "categorical",
            categorical_transformer,
            make_column_selector(dtype_include="category"),
        ),
        (
            "numerical",
            numerical_transformer,
            make_column_selector(dtype_include=np.number),
        ),
    ],
    remainder="drop",
    verbose=True,
)

ord_preprocessing_pipeline = ColumnTransformer(
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
    verbose=True,
)
# %%
# %%
logreg_rfe_pipeline = Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("Feature Selection", rfe_lr),
        ("LogReg_Classifier", lr),
    ]
)

logreg_rfe_smote_pipeline = imb_pipeline.Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("SMOTE oversampling", smt),
        ("Feature Selection", rfe_lr),
        ("LogReg_Classifier", lr),
    ]
)
#%%
dt_rfe_pipeline = Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("Feature Selection", rfe_dt),
        ("LogReg_Classifier", dt),
    ]
)

dt_rfe_smote_pipeline = imb_pipeline.Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("SMOTE oversampling", smt),
        ("Feature Selection", rfe_dt),
        ("LogReg_Classifier", dt),
    ]
)

#%%
rfc_rfe_pipeline = Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("Feature Selection", rfe_rfc),
        ("LogReg_Classifier", rfc),
    ]
)

rfc_rfe_smote_pipeline = imb_pipeline.Pipeline(
    [
        ("Preprocessing Step", preprocessing_pipeline),
        ("SMOTE oversampling", smt),
        ("Feature Selection", rfe_rfc),
        ("LogReg_Classifier", rfc),
    ]
)

#%%


# %%
dt_pipeline = Pipeline(
    [("Preprocessing Step", preprocessing_pipeline), ("DT_Classifier", dt),]
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
