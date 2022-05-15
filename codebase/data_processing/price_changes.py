# %% Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from processing import M5Data, plot_sales_price

# %% Get data

m5data = M5Data(day_start=1)

# %% Look at price changes

prices = m5data.prices
print(f"Unique Entries:\n{prices.nunique()}")
# Only 1048 unique price values

# %% List of items with more than one price, at the same store

item_prices = prices.groupby(["item_id", "store_id"])["sell_price"].agg(
    ["nunique", "mean", "max", "min", "var"]
)
item_prices["nunique"].value_counts().head()
# Most items have only had 1 or 2 price changes


# %% Has an item every had different prices at the same time?

item_weekly_prices = prices.groupby(["item_id", "wm_yr_wk"])["sell_price"].nunique()
item_weekly_prices.value_counts().head()
# Prices are not uniform across stores...
# Cross store interactions will be difficult

# %%

item_price_time = prices.groupby(["item_id", "store_id", "wm_yr_wk"])[
    "sell_price"
].mean()

price_changes = (item_price_time.diff() != 0).astype(int).reset_index()
changing_products = (price_changes.groupby("item_id")["sell_price"].sum()).clip(lower=0)

# %%

fig, ax = plt.subplots(figsize=(9, 4.5))
(changing_products / 10).hist(bins=20, ax=ax)
ax.set(
    title="Histogram of Number of Price Changes per Product per Store",
    xlabel="Number of Price Changes",
    ylabel="Number of Products",
)

common_changes = changing_products[changing_products > 34].index

# %% Look at some individual items...

expensive_varying = item_prices.query("nunique > 4 and var > 0.03 and mean > 5.80")
cheap_varying = item_prices.query("nunique > 4 and var > 0.03 and mean < 5.80")


# %% Find some products with frequent price changes
# %% Find some products which is sold at high volumes

for aitem, astore in cheap_varying.sample(20).index:
    if aitem not in common_changes:
        continue
    if "FOODS" not in aitem:
        continue
    print(aitem, astore)
    m5data.make_subset(f"item_id == '{aitem}' and store_id == '{astore}'")
    df = m5data.subset
    assert not df.empty
    dfw = df.resample("W").agg({"demand": np.sum, "sell_price": np.mean})
    plot_sales_price(dfw, title=f"{aitem} AT {astore}")

# %% Make the processed data
print("Making subset")
candidate = "FOODS_2_298"
candidate = "FOODS_2_183"
candidate = "FOODS_1_093"
candidate = "FOODS_1_096"
candidate = "FOODS_1_199"
candidate = "FOODS_2_173"
candidate = "FOODS_3_174"

candidate = "FOODS_3_578"
candidate = "FOODS_2_058"
candidate = "FOODS_2_373"
candidate = "FOODS_3_219"  ## promising but big zero period
candidate = "FOODS_2_373"
candidate = "FOODS_1_024"


#%%

list_candidates = [
    "FOODS_1_024",
    "FOODS_1_093",
    "FOODS_1_096",
    "FOODS_1_199",
    "FOODS_2_058",
    "FOODS_2_173",
    "FOODS_2_183",
    "FOODS_2_298",
    "FOODS_2_373",
    "FOODS_3_174",
    "FOODS_3_578",
]

candidate = list_candidates[-1]

"""
Some candidates, 1_096 selected
"""
all_candidates = []

for candidate in list_candidates:
    m5data.make_subset(f"item_id == '{candidate}'")
    df = m5data.subset

    dfout = (
        df.groupby("store_id")
        .resample("W")
        .agg({"demand": np.sum, "sell_price": np.mean})
        .reset_index()
    )

    dfout["ID"] = candidate
    all_candidates.append(dfout)


#%%

pd.concat(all_candidates, ignore_index=True).to_csv("multi_itme_weekly.csv")
#%%

"""
saved as processed data

"""

fig, ax = plt.subplots(2, 1, figsize=(15, 10))
sns.lineplot(data=dfout, y="demand", x="date", hue="store_id", ax=ax[0])
sns.lineplot(data=dfout, y="sell_price", x="date", hue="store_id", ax=ax[1])

# dfout.to_csv('test_item_weekly.csv', index = False)


#%%


#%%


# %% Check plots
# dfout = dfout.reset_index()
# dfout["ds"] = pd.to_datetime(dfout["date"])


# fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# sns.lineplot(x="ds", y="demand", hue="store_id", data=dfout, ax=ax[0])
# sns.lineplot(x="ds", y="sell_price", hue="store_id", data=dfout, ax=ax[1])
# ax[0].set(title=f'{candidate} Weekly Sales and Price Changes')


# %%

# %%
