#%%
from typing import Tuple, List
import pandas as pd
import numpy as np

#%%
import json

import requests
from ExtractData import FetchSubset

#%%
import os

cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)
#%%
with open("../config.json", "r") as f:
    config = json.load(f)
#%%
class ProviderNumCategoryCreator(object):
    """Class to classify the PRVDR_NUM column data into appropriate category
    Pass the dataframe / series object to the get_categories_for_providers method
    """

    def __init__(self) -> None:
        self.unique_prvdr_num_category_df = pd.DataFrame()
        self.unique_prvdr_num_df = pd.DataFrame()
        self.current_col_data = pd.Series()
        self.prvdr_num_bins = self._getProviderNumGroups()

    def _getProviderNumGroups(self) -> List:
        prvdr_num_categories = pd.read_csv(config[1]["prvdr_category_file"])
        prvdr_num_bins = [
            cat
            for cat in prvdr_num_categories["PRVDR_CAT"].str.strip().str.split("-")
            if len(cat) > 1
        ]

        return prvdr_num_bins

    def getUniqueProviders(self):
        if self.unique_prvdr_num_df.empty:
            self.unique_prvdr_num_df = pd.DataFrame(
                self.current_col_data.unique(), columns=["PRVDR_NUM"]
            )
        else:
            temp_series = pd.concat(
                [self.unique_prvdr_num_df, self.current_col_data], axis=0
            )
            self.unique_prvdr_num_df = pd.DataFrame(
                temp_series[0].unique(), columns=["PRVDR_NUM"]
            )
        self.unique_prvdr_num_category_df["PRVDR_NUM"] = self.unique_prvdr_num_df[
            "PRVDR_NUM"
        ]

    def find_prvdr_num_category(self, prvdr_num: str):
        if not prvdr_num:
            return np.nan
        if str(prvdr_num)[2].isalpha():
            return str(prvdr_num)[2]
        else:
            for lower, upper in self.prvdr_num_bins:
                if lower <= prvdr_num[:-2] <= upper:
                    return f"{lower}-{upper}"

    def get_categories_for_providers(self, provider_column: pd.Series):
        """Method to call for categorizing the PRVR_NUM related columns

        Args:
            provider_column (pd.Series): [PRVDR column of dataframe]
        """
        self.current_col_data = provider_column
        self.getUniqueProviders()
        self.unique_prvdr_num_category_df[
            "PRVDR_NUM_CAT"
        ] = self.unique_prvdr_num_category_df["PRVDR_NUM"].apply(
            lambda x: self.find_prvdr_num_category(x)
        )


#%%
class DiagnosisCodeCategoryCreator(object):
    """Class to classify the ICD9 Diagnosis Code column data into appropriate category
    Pass the dataframe object to the get_categories_for_providers method
    """

    def __init__(self) -> None:
        self.unique_diagnosis_code_category_df = pd.DataFrame()
        self.unique_diagnosis_code_df = pd.DataFrame()
        self.current_diagnosis_df = pd.DataFrame()
        self.diagnosis_code_bin = self._getICDDiagnosisCodeCategory()

    def _getICDDiagnosisCodeCategory(self) -> List[Tuple[str, str]]:
        response = requests.get(
            config[2]["ICD9_DIAGNOSIS_CODE_CATEGORY_URL"], verify=False
        )
        icd_code_categories = pd.read_html(response.text)[0]
        icd_diagnosis_code_bin = [
            x for x in icd_code_categories["Code Range"].str.split("-")
        ]
        return icd_diagnosis_code_bin

    def getUniqueDiagnosisCodes(self):
        for col in self.current_diagnosis_df:
            self.unique_diagnosis_code_df = pd.concat(
                [
                    self.unique_diagnosis_code_df,
                    pd.DataFrame(self.current_diagnosis_df[col].values),
                ],
                axis=0,
            )

        self.unique_diagnosis_code_category_df = pd.DataFrame(
            self.unique_diagnosis_code_df[0].unique(), columns=["Diagnosis_code"]
        )
        self.unique_diagnosis_code_category_df[
            "Diagnosis_code"
        ] = self.unique_diagnosis_code_category_df["Diagnosis_code"].str[:3]
        self.unique_diagnosis_code_category_df["Diagnosis_code"].astype(str).replace(
            "nan", np.nan
        )

    def find_diagnosis_code_category(self, diagnosis_code: str):
        if pd.isnull(diagnosis_code):
            return np.nan
        elif "E" in str(diagnosis_code):
            return "E000-E999"
        elif "V" in str(diagnosis_code):
            return "V01-V91"
        elif "OTH" in str(diagnosis_code):
            return "OTHER"
        else:
            for lower, upper in self.diagnosis_code_bin[:-2]:
                if all(
                    [diagnosis_code.isnumeric(), lower.isnumeric(), upper.isnumeric(),]
                ) & (lower <= diagnosis_code <= upper):
                    return f"{lower}-{upper}"

    def get_categories_for_diagnosis_code(self, diagnosis_data: pd.DataFrame):
        """Method to call for categorizing the ICD9 Diagnosis Code

        Args:
            diagnosis_data (pd.DataFrame): [DataFrame of ICD9 Diagnosis Code]
        """
        self.current_diagnosis_df = diagnosis_data
        self.getUniqueDiagnosisCodes()
        self.unique_diagnosis_code_category_df[
            "Diagnosis_code_CAT"
        ] = self.unique_diagnosis_code_category_df["Diagnosis_code"].apply(
            self.find_diagnosis_code_category
        )
        self.unique_diagnosis_code_category_df.drop_duplicates(inplace=True)


#%%
class ProcedureCodeCategoryCreator(object):
    """
    Class to classify the ICD9 Procedure code data into appropriate category
    Pass the dataframe / series object to the get_categories_for_providers method
    """

    def __init__(self) -> None:
        self.unique_procedure_code_category_df = pd.DataFrame()
        self.unique_procedure_code_df = pd.DataFrame()
        self.current_procedure_df = pd.DataFrame()
        self.icd_procedure_code_bin = self._getICDProcedureCodeCategory()

    def _getICDProcedureCodeCategory(self) -> List[Tuple[str, str]]:
        procedural_code_mapping = pd.read_csv(
            config[1]["icd9_procedure_code_category_file"], encoding="utf-8"
        )

        icd_procedure_code_bin = [
            x for x in procedural_code_mapping["Code Range"].str.strip().str.split("-")
        ]
        return icd_procedure_code_bin

    def getUniqueProcedureCodes(self):
        for col in self.current_procedure_df:
            self.unique_procedure_code_df = pd.concat(
                [
                    self.unique_procedure_code_df,
                    pd.DataFrame(self.current_procedure_df[col].values),
                ],
                axis=0,
            )

        self.unique_procedure_code_category_df = pd.DataFrame(
            self.unique_procedure_code_df[0].unique(), columns=["Procedure_code"]
        )
        self.unique_procedure_code_category_df.loc[
            :, "Procedure_code"
        ] = self.unique_procedure_code_category_df.loc[:, "Procedure_code"].astype(
            "str"
        )

        self.unique_procedure_code_category_df[
            "Procedure_code"
        ] = self.unique_procedure_code_category_df["Procedure_code"].str[:2]

        self.unique_procedure_code_category_df["Procedure_code"].astype(str).replace(
            "na", np.nan
        )

    def find_procedure_code_category(self, procedure_code: str):
        if pd.isnull(procedure_code):
            return np.nan
        else:
            for lower, upper in self.icd_procedure_code_bin:
                if lower <= procedure_code <= upper:
                    return f"{lower}-{upper}"

    def get_categories_for_procedure_code(self, procedure_data: pd.DataFrame):
        """Method to call for categorizing the ICD9 Procedure Code

        Args:
            diagnosis_data (pd.DataFrame): [DataFrame of ICD9 Procedure Code]
        """
        self.current_procedure_df = procedure_data
        self.getUniqueProcedureCodes()
        self.unique_procedure_code_category_df[
            "Procedure_code_CAT"
        ] = self.unique_procedure_code_category_df["Procedure_code"].apply(
            self.find_procedure_code_category
        )
        self.unique_procedure_code_category_df.drop_duplicates(inplace=True)


#%%
class HCPCSCodeCategoryCreator(object):
    """
    Class to classify the ICD9 Procedure code data into appropriate category
    Pass the dataframe / series object to the get_categories_for_providers method.

    There may be some old HCPCS code in the data which can be replaced with OLD_HCPCS_CODE_MAPPING dict.
    """

    OLD_HCPCS_CODE_MAPPING = {
        "90772": "96372",
        "78006": "78014",
        "78007": "78014",
        "78010": "78013",
        "78011": "78013",
        "80100": "G0431",
        "80101": "G0431",
        "80104": "G0434",
        "78000": "78012",
        "78001": "78012",
        "78003": "78012",
        "78010": "78013",
        "78011": "78013",
        "78006": "78014",
        "78007": "78014",
        "10022": "10004",
        "96101": "96130",
        "96102": "96138",
        "82003": "80329",
        "80100": "G0430",
        "80101": "G0431",
        "82003": "G6039",
        "77031": "19081",
        "77032": "19082",
        "99144": "99152",
        "93875": "93880",
        "90921": "G0308",
        "99143": "99151",
        "99143": "99151",
        "99144": "99152",
        "99145": "99153",
        "99148": "99155",
        "99149": "99156",
        "99150": "99157",
        "0187T": "92132",
        "0030T": "86849",
    }

    def __init__(self) -> None:
        self.unique_hcpcs_code_category_df = pd.DataFrame()
        self.unique_hcpcs_code_df = pd.DataFrame()
        self.current_hcpcs_df = pd.DataFrame()
        (
            self.hcpcs_code_category_numeric,
            self.hcpcs_code_category_alnum,
        ) = self._getICDHCPCSCodeCategory()

    def _getICDHCPCSCodeCategory(self) -> Tuple[List, List]:
        hcpcs_code_category = pd.read_csv(
            config[1]["hcpcs_code_category_file"], encoding="cp1252"
        )

        hcpcs_code_category_numeric = [
            x
            for x in hcpcs_code_category["Code Range"].str.split("-")
            if x[0].isnumeric()
        ]
        hcpcs_code_category_alnum = [
            x
            for x in hcpcs_code_category["Code Range"].str.split("-")
            if x not in hcpcs_code_category_numeric
        ]

        return hcpcs_code_category_numeric, hcpcs_code_category_alnum

    def getUniqueHCPCSCodes(self):
        for col in self.current_hcpcs_df:
            self.unique_hcpcs_code_df = pd.concat(
                [
                    self.unique_hcpcs_code_df,
                    pd.DataFrame(self.current_hcpcs_df[col].values),
                ],
                axis=0,
            )

        self.unique_hcpcs_code_category_df = pd.DataFrame(
            self.unique_hcpcs_code_df[0].unique(), columns=["HCPCS_code"]
        )

    def _categorize_hcpcs_code_2(self, modifier, hcpcs_code):
        categories = [
            [x[0][:-1], x[1][:-1]]
            for x in self.hcpcs_code_category_alnum
            if modifier in x[0]
        ]
        for lower, upper in categories:
            if lower <= hcpcs_code <= upper:
                return f"{lower}{modifier}-{upper}{modifier}"

    def find_hcpcs_code_category(self, hcpcs_code: str):
        if pd.isnull(hcpcs_code):
            return np.nan
        elif str(hcpcs_code)[0].isalpha():
            return str(hcpcs_code)[0]
        elif str(hcpcs_code)[-1].isalpha():
            return self._categorize_hcpcs_code_2(
                str(hcpcs_code)[-1], str(hcpcs_code)[:-1]
            )
        elif str(hcpcs_code)[0].isnumeric():
            for lower, upper in self.hcpcs_code_category_numeric:
                if lower <= hcpcs_code <= upper:
                    return f"{lower}-{upper}"

    def get_categories_for_hcpcs_code(self, hcpcs_data: pd.DataFrame):
        """
        Method to call for categorizing the HCPCS Code
        There may be some old HCPCS code in the data which should be replaced with OLD_HCPCS_CODE_MAPPING dict.

        Args:
            diagnosis_data (pd.DataFrame): [DataFrame of HCPCS Code]
        """
        self.current_hcpcs_df = hcpcs_data
        self.getUniqueHCPCSCodes()
        self.unique_hcpcs_code_category_df[
            "HCPCS_code_CAT"
        ] = self.unique_hcpcs_code_category_df["HCPCS_code"].apply(
            self.find_hcpcs_code_category
        )
        self.unique_hcpcs_code_category_df.drop_duplicates(inplace=True)


#%%
if __name__ == "__main__":
    # inpatient_file = FetchSubset(subset_List=[2]).fetchFromInpatientDataset()
    # outpatient_file = FetchSubset(subset_List=[2]).fetchFromOutpatientDataset()
    create_provider_category = ProviderNumCategoryCreator()
    diagnosis_code = DiagnosisCodeCategoryCreator()
    procedure_code = ProcedureCodeCategoryCreator()
    hcpcs_code = HCPCSCodeCategoryCreator()
    print("Import & Use")
