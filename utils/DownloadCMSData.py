#%%
import requests

#%%
import os

cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)

#%%
def get_beneficiary_summary_file(sample=1, year=2008):
    URL = f"https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/DE1_0_{year}_Beneficiary_Summary_File_Sample_{sample}.zip"
    if not os.path.exists(f"../input/DE1.0 Sample{sample}"):
        os.mkdir(f"../input/DE1.0 Sample{sample}")
    try:
        response = requests.get(URL, stream=True, allow_redirects=True, verify=False)
        with open(
            f"../input/DE1.0 Sample{sample}/DE1_0_{year}_Beneficiary_Summary_File_Sample_{sample}.zip",
            "wb",
        ) as handle:
            for chunk in response.iter_content(chunk_size=512):
                if chunk:  # filter out keep-alive new chunks
                    handle.write(chunk)
    except Exception as e:
        print(
            f"Unable to download beneficiary summary files for DE_{sample:2d} sample of year {year}. The exception message received is {e} "
        )


def get_carrier_claims(sample=1, claim_type="A"):
    URL = f"http://downloads.cms.gov/files/DE1_0_2008_to_2010_Carrier_Claims_Sample_{sample}{claim_type}.zip"

    if not os.path.exists(f"../input/DE1.0 Sample{sample}"):
        os.mkdir(f"../input/DE1.0 Sample{sample}")
    try:
        response = requests.get(URL, stream=True, allow_redirects=True, verify=False)
        with open(
            f"../input/DE1.0 Sample{sample}/DE1_0_2008_to_2010_Carrier_Claims_Sample_{sample}{claim_type}.zip",
            "wb",
        ) as handle:
            for chunk in response.iter_content(chunk_size=512):
                if chunk:  # filter out keep-alive new chunks
                    handle.write(chunk)
    except Exception as e:
        print(
            f"Unable to download Carrier claims file for DE_{sample:2d} claimtype {claim_type}. The exception received is {e} "
        )


def get_prescription_drug_events(sample=1):
    URL = f"http://downloads.cms.gov/files/DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_{sample}.zip"

    if not os.path.exists(f"../input/DE1.0 Sample{sample}"):
        os.mkdir(f"../input/DE1.0 Sample{sample}")
    try:
        response = requests.get(URL, stream=True, allow_redirects=True, verify=False)
        with open(
            f"../input/DE1.0 Sample{sample}/DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_{sample}.zip",
            "wb",
        ) as handle:
            for chunk in response.iter_content(chunk_size=512):
                if chunk:  # filter out keep-alive new chunks
                    handle.write(chunk)
    except Exception as e:
        print(
            f"Unable to download Prescription Drug Event file for DE_{sample:2d}. The exception received is {e} "
        )


def get_patient_care_claims(sample=1, care_type="Inpatient"):
    URL = f"https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/DE1_0_2008_to_2010_{care_type}_Claims_Sample_{sample}.zip"

    if not os.path.exists(f"../input/DE1.0 Sample{sample}"):
        os.mkdir(f"../input/DE1.0 Sample{sample}")
    try:
        response = requests.get(URL, stream=True, allow_redirects=True, verify=False)
        with open(
            f"../input/DE1.0 Sample{sample}/DE1_0_2008_to_2010_{care_type}_Claims_Sample_{sample}.zip",
            "wb",
        ) as handle:
            for chunk in response.iter_content(chunk_size=512):
                if chunk:  # filter out keep-alive new chunks
                    handle.write(chunk)
    except Exception as e:
        print(
            f"Unable to {care_type} claims file for DE_{sample:2d}. The exception received is {e} "
        )


# %%
if __name__ == "__main__":
    for i in range(1, 21):
        # download beneficiary summary files
        get_beneficiary_summary_file(sample=i, year=2008)
        get_beneficiary_summary_file(sample=i, year=2009)
        get_beneficiary_summary_file(sample=i, year=2010)

        # download carrier claims files
        get_carrier_claims(sample=i, claim_type="A")
        get_carrier_claims(sample=i, claim_type="B")

        # download prescription drug event file
        get_prescription_drug_events(sample=i)

        # download Inpatient and Outpatient files
        get_patient_care_claims(sample=i, care_type="Inpatient")
        get_patient_care_claims(sample=i, care_type="Outpatient")

# %%
