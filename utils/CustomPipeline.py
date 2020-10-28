#%%
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import json

#%%
with open("../config.json", "r") as f:
    config = json.load(f)

#%%
HIGH_CARDINALITY_THRESHOLD = config[3]["HIGH_CARDINALITY_THRESHOLD"]
DEFAULT_INFREQUENT_CATEGORY_CUT_OFF = config[3]["DEFAULT_INFREQUENT_CATEGORY_CUT_OFF"]
DEFAULT_INFREQUENT_CATEGORY_LABEL = config[3]["DEFAULT_INFREQUENT_CATEGORY_LABEL"]
#%%
class CategoricalVariableImputer(BaseEstimator, TransformerMixin):
    """Class to fill the missing values in Categorical Data

        Args:
            fill_value (str, optional): [description]. Defaults to "UNKNOWN".
        """

    def __init__(self, fill_value: str = "UNKNOWN") -> None:
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None):
        """
        docstring
        """
        return self

    def transform(self, X: pd.DataFrame, y=None):
        print(f"{__class__} Transform Method Called")
        assert isinstance(X, pd.DataFrame)
        for col in X:
            X[col] = X[col].cat.add_categories(self.fill_value)
            X[col] = X[col].fillna(value=self.fill_value)

        return X


#%%
class CardinalityReducer(BaseEstimator, TransformerMixin):
    """Class to reduce Cardinality of the Categorical Columns

    Args:
        cut_off ([int]): [Minimum frequency below which the column is considered as infrequent]
        label ([String]): [Name of the category which would combine the low frequency columns]

    """

    # Class Constructor
    def __init__(
        self,
        cutt_off: int = DEFAULT_INFREQUENT_CATEGORY_CUT_OFF,
        label: str = DEFAULT_INFREQUENT_CATEGORY_LABEL,
    ):
        print(f"{__class__} Initialized")

        self.cutt_off = cutt_off
        self.label = label
        self.column_category = dict()

    # Return self nothing else to do here
    def fit(self, X, y=None):
        print(f"{__class__} fit method called")
        data = pd.DataFrame(X)
        for col in data:
            isHighCardinal = data[col].nunique() > HIGH_CARDINALITY_THRESHOLD
            if isHighCardinal:
                self.column_category[col] = self.getInfrequentCategories(data[col])

        print(f"{__class__} Fit method executed")
        return self

    # Helper method to get categories which will be replaced with others
    def getInfrequentCategories(self, X):
        """Method to get the list of all the categories in the given feature that hvae frequenct equal to less than cut_off value

        Args:
            X ([Series]): [Feature with high cardinality]

        Returns:
            [index]: [list of categories with low frequency]
        """
        categories_frequency = X.value_counts()
        infrequent_categories = categories_frequency[
            categories_frequency <= self.cutt_off
        ].index
        return list(infrequent_categories)

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        """Merge all the categories with low frequency into a single category

        Args:
            X ([DataFrame]): [DataFrame of Categorical features]
            y ([type], optional): [description]. Defaults to None.

        Returns:
            [DataFrame]: [DataFrame with all the low frequency categories merged into one category]
        """
        print(f"{__class__} Transform method called")

        data = pd.DataFrame(X).astype("category")
        for col, infrequent_category in self.column_category.items():
            data[col] = data[col].cat.add_categories([self.label])
            data[col] = data[col].replace(infrequent_category, self.label)

        print(f"{__class__} Transform method executed")
        return data


def get_feature_out(estimator, feature_in):
    if hasattr(estimator, "get_feature_names"):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f"vec_{f}" for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


class SelectColumnsTransfomer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection
    
    Allows to select or drop columns by name from pandas dataframes in scikit-learn
    pipelines.
    
    Parameters
    ----------
    columns (List) : list of str, names of the dataframe columns to select
    drop (bool, optional): [description]. Defaults to False.

    """

    def __init__(self, columns: List, drop: bool = False):
        print(f"{__class__} Initialized")
        self.columns = columns
        self.drop = drop

    def transform(self, X: pd.DataFrame, **transform_params):
        """ Selects or drop columns of a DataFrame.
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
            contains selected columns of X      
        """
        print(f"{__class__} Transform method called")
        if self.columns:
            if self.drop:
                trans = X.drop(columns=self.columns)
            else:
                trans = X[self.columns].copy()
        else:
            return X
        return trans

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        if self.columns:
            self.columns = [col for col in self.columns if col in X.columns]

        print(f"Columns {self.columns} will be dropped")
        return self


def get_ct_feature_names(ct: ColumnTransformer):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name != "remainder":
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator == "passthrough":
            output_features.extend(ct._feature_names_in[features])

    return output_features


#%%
if __name__ == "__main__":
    print("Import and use")
