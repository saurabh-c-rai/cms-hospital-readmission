#%%
import os
import requests

#%%
class ICD9Codes:
    def __init__(self):
        self.URL = "https://icd.codes/api/search"
        self.HEADERS = {
            "Cookie": "__cfduid=d4382452363aff5d8718966c49150ddae1600517456; XSRF-TOKEN=eyJpdiI6InF4Nmx5ejVoRVI5V0dLMHF3VjlPa2c9PSIsInZhbHVlIjoiWWIyTFN2XC9IK1hyWExDaTBFNlczcjR2R1VJZ0MydG1kaUJJK0ZpcG51anZFN3JZMUJnK3VKTjlWbDFjakJmUWN5amVoTTd1U3l2T2ZoZ0NOMHYrREtBPT0iLCJtYWMiOiJjOTc0OTViNDExOTRjN2U1YTZhODM2ZGQ1Y2Q0MWZlNDdhNzc1MGRjNTViZTUxNDA0MWU3ZmViNGE1ZDhiN2E0In0%3D; laravel_session=eyJpdiI6IjVWaW0rRWdCWEtyNmtMSERPdkI4MEE9PSIsInZhbHVlIjoiR0JIUzRGeUw3cXQzQUVwQ3dyNjV6M014bktDK0dVMjFcLyt6VE43M1poanplWjJudVhoUUJvRVwvXC9wa2hseTdPV0JaVHYwWW1YMERLelwvOG1zdk5adGd3PT0iLCJtYWMiOiI2NmVjMDcxMjI3M2U5ODM3Mjc1NTU2MThiZTdmOTEwNGRhNjQ3NmFhZGM0ZTJkMzg2NmZmNGY1YWQ2NmUwMWE0In0%3D"
        }
        self.FILES = []
        self.method = "POST"

    def create_request(self):
        request_data = {
            "header": self.HEADERS,
            "files": self.FILES,
        }

        return request_data

    def get_icd_code_details(self, codeType, icdCode):
        payload = {
            "type": codeType,
            "query": "standard",
            "term": icdCode,
            "size": "1",
            "from": "0",
        }

        # request_object = self.create_request()
        response = requests.request(
            url=self.URL,
            method=self.method,
            headers=self.HEADERS,
            files=self.FILES,
            data=payload,
            verify=False,
        )
        return response.json()

    def get_description_for_icd_code(self, ICD9Codes, codeType="icd9cm"):
        try:
            response = self.get_icd_code_details(codeType=codeType, icdCode=ICD9Codes)
            return response["hits"]["hits"][0]["_source"]["desc"]
        except Exception as e:
            print(f"Failed for code {ICD9Codes} for codetype {codeType}")
            return "Not Found"

    # def generate_json_file(self, *kwarg):
    #     icdcode = kwarg["icdCode"]
    #     description = kwarg["desc"]
    #     if not os.path.exists(f"../icd9mapping.json"):
    #         os.mkdir(f"../icd9mapping.json")
    #     with open("../icd9mapping.json", "a") as target:
    #         pass


#%%
if __name__ == "__main__":
    fetch_data = ICD9Codes()
    print(fetch_data.get_description_for_icd_code(codeType="icd9cm", ICD9Codes="2859"))
