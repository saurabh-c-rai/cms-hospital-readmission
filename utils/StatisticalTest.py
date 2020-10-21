#%%
import numpy as np
import scipy.stats as ss
import pandas as pd
import json

#%%
with open("../config.json", "r") as f:
    config = json.load(f)

#%%
STATISTICAL_TEST_ALPHA = config[3]["STATISTICAL_TEST_ALPHA"]

#%%
class ChiSquare:
    def __init__(self, df):
        self.df = df
        self.p = None  # P-Value
        self.chi2 = None  # Chi Test Statistic
        self.dof = None
        self.dfObserved = None
        self.dfExpected = None
        self.dfReleventCols = []
        self.dfIrreleventCols = []

    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p < alpha:
            result = f"{colX} is IMPORTANT for Prediction"
            self.dfReleventCols.append(colX)

        else:
            result = (
                f"{colX} is NOT an important predictor. (Discard {colX} from model)"
            )
            self.dfIrreleventCols.append(colX)

        print(result)

    def TestIndependence(self, colX, colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)

        self.dfObserved = pd.crosstab(Y, X)
        chi2, p, dof, expected = ss.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof
        self.dfExpected = pd.DataFrame(
            expected, columns=self.dfObserved.columns, index=self.dfObserved.index
        )
        self._print_chisquare_result(colX, alpha)

    def cramers_v(self, colX, colY):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        self.dfObserved = pd.crosstab(X, Y)
        self.chi2 = ss.chi2_contingency(self.dfObserved)[0]
        n = self.dfObserved.sum().sum()
        phi2 = self.chi2 / n
        r, k = self.dfObserved.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        corr_coef = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
        return corr_coef


#%%
if __name__ == "__main__":
    pass

