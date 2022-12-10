import pytest
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
from pcregression import pcregression as pcr


@pytest.fixture
def create_test_data():
    # creating data
    data = pd.read_csv("islr_college_test_data.csv")
    X = StandardScaler().fit_transform(data.drop(["Apps"], axis=1))
    y = StandardScaler().fit_transform(data.Apps.to_numpy().reshape(-1, 1))

    # pca with 2 components
    pca = PCA(n_components=2).fit(X)
    xpca = pca.transform(X)
    evr = pca.explained_variance_ratio_

    return {"X": X, "y": y, "xpca": xpca, "explained_variance": evr}


class Test_PCA_Regression:
    def test_reg(self, create_test_data):
        self.X = create_test_data["X"]
        self.y = create_test_data["y"]
        self.xpca = create_test_data["xpca"]
        self.evr = create_test_data["explained_variance"]

        lm = LinearRegression().fit(self.xpca, self.y)
        lm_preds = lm.predict(self.xpca)
        pcar = pcr.PCARegression(n_components=2)
        pcar.fit(self.X, self.y)
        pcar_preds = pcar.predict(self.X)
        print(pcar_preds[:5])
        print(lm_preds[:5])

        # test1: assert explained variance of first 2 components is equal
        assert (
            np.around(self.evr, 3) == np.around(pcar.pca.explained_variance_ratio_, 3)
        ).all() == True

        # test2: assert predictions are equal from both regressions
        assert (np.around(lm_preds, 2) == np.around(pcar_preds, 2)).all() == True
