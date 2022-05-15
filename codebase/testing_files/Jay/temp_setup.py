#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from sklearn.metrics import mean_absolute_percentage_error

# %%


def sMAPE(actual, forecast):
    multiplier = 100 / len(actual)
    nominator = np.array(2 * np.abs(forecast - actual)).astype(float)
    denominator = np.array(np.abs(actual) + np.abs(forecast)).astype(float)

    return multiplier * np.sum(
        np.divide(
            nominator, denominator, out=np.zeros_like(nominator), where=denominator != 0
        )
    )


# %%
dfin = pd.read_csv("../processed_data/test_item_weekly.csv")

dfin["ds"] = pd.to_datetime(dfin["date"])
dfin["y"] = dfin["demand"]
dfin["xp"] = dfin["sell_price"]

dfin["state"] = dfin["store_id"].str[:2]
dfin["store"] = dfin["store_id"].str[-1:]
dfin.head()
