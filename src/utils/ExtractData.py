#%%
import random
import pandas as pd
import numpy as np

#%%
import os

cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)

# %%
class FetchSubset:
    """
    This class will be used to fetch data from csv files.
    Args:
            no_of_subset ([int]): [total no. of subset from which data will be fetched]
            subset_list ([list]): [list of subset no. to fetch from particular subset]
    """

    def __init__(self, no_of_subset: int = None, subset_list: list = []):
        """
        Constructor for the Fetch Subset object

        """
        self.no_of_subset = no_of_subset
        self.subset_list = subset_list
        self.getSubsetIndex()

    def getSubsetIndex(self, **parameter_list: dict):
        """
        This method will generate the index if no_of subset is passed or will pass the subset_list if no_of_subset is None
        """
        if not self.subset_list:
            self.subset_list = random.sample(range(2, 21), self.no_of_subset)
        return self.subset_list

    def fetchFromInpatientDataset(self) -> pd.DataFrame:
        """This method will read the required data from inpatient file and concatinate them and return the dataframe

        Returns:
            dataFrame: [inpatient data]
        """
        dataframe_list = []
        for i in self.subset_list:
            data_inpatient_claims = pd.read_csv(
                f"..\input\DE1.0 Sample{i}\DE1_0_2008_to_2010_Inpatient_Claims_Sample_{i}.zip",
                parse_dates=[
                    "CLM_FROM_DT",
                    "CLM_THRU_DT",
                    "CLM_ADMSN_DT",
                    "NCH_BENE_DSCHRG_DT",
                ],
                infer_datetime_format=True,
            )
            dataframe_list.append(data_inpatient_claims)

        final_inpatient_data = pd.concat(dataframe_list, axis=0)

        return final_inpatient_data

    def fetchFromPrescriptionDrugEventsDataset(self) -> pd.DataFrame:
        """This method will read the required data from Prescription Drug Events file and concatinate them and return the dataframe

        Returns:
            dataFrame: [prescription drug event data]
        """
        dataframe_list = []
        for i in self.subset_list:
            data_prescription_drug_event = pd.read_csv(
                f"..\input\DE1.0 Sample{i}\DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_{i}.zip",
                parse_dates=["SRVC_DT",],
                infer_datetime_format=True,
            )
            dataframe_list.append(data_prescription_drug_event)

        final_prescription_drug_event = pd.concat(dataframe_list, axis=0)

        return final_prescription_drug_event

    def fetchFromOutpatientDataset(self) -> pd.DataFrame:
        """This method will read the required data from outpatient file and concatinate them and return the dataframe

        Returns:
            dataFrame: [outpatient data]
        """
        dataframe_list = []
        for i in self.subset_list:
            data_outpatient_claims = pd.read_csv(
                f"..\input\DE1.0 Sample{i}\DE1_0_2008_to_2010_Outpatient_Claims_Sample_{i}.zip",
                parse_dates=["CLM_FROM_DT", "CLM_THRU_DT",],
                infer_datetime_format=True,
            )
            dataframe_list.append(data_outpatient_claims)

        final_outpatient_data = pd.concat(dataframe_list, axis=0)

        return final_outpatient_data

    def fetchFromBeneficiaryDataset(self, year: int = 2008) -> pd.DataFrame:
        """This method will read the required data from beneficiary summary file and concatinate them and return the dataframe

        Args:
            year (int, optional): [Report Year]. Defaults to 2008.

        Returns:
            dataFrame: [beneficiary data for the required year]
        """
        assert year in [2008, 2009, 2010], "Incorrect Year Given"
        dataframe_list = []
        for i in self.subset_list:
            data_beneficiary_summary = pd.read_csv(
                f"..\input\DE1.0 Sample{i}\DE1_0_{year}_Beneficiary_Summary_File_Sample_{i}.zip",
                parse_dates=["BENE_BIRTH_DT", "BENE_DEATH_DT",],
                infer_datetime_format=True,
            )
            dataframe_list.append(data_beneficiary_summary)

        final_beneficiary_data = pd.concat(dataframe_list, axis=0)

        return final_beneficiary_data

    def fetchFromCarrierClaimsDataset(self, claim_type: str = "A") -> pd.DataFrame:
        """This method will read the required data from carrier claims file and concatinate them and return the dataframe


        Args:
            claim_type (str, optional): [claim type]. Defaults to "A".

        Returns:
            dataFrame: [carrier claim data for the required claim type]
        """
        assert claim_type in ["A", "B"], "Incorrect Claim Type Given"
        dataframe_list = []
        for i in self.subset_list:
            data_carrier_claims = pd.read_csv(
                f"..\input\DE1.0 Sample{i}\DE1_0_2008_to_2010_Carrier_Claims_Sample_{i}{claim_type}.zip",
                parse_dates=["CLM_FROM_DT", "CLM_THRU_DT",],
                infer_datetime_format=True,
            )
            dataframe_list.append(data_carrier_claims)

        final_carrier_claims_data = pd.concat(dataframe_list, axis=0)

        return final_carrier_claims_data


if __name__ == "__main__":
    # data = FetchSubset(no_of_subset=5)
    # inpatient_data = data.fetchFromBeneficiaryDataset()
    # print(inpatient_data.shape)
    print("Import and use the fetch data")
    pass

