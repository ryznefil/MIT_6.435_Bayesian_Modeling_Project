#%%
from temp_setup import *

## not ideal, at least make into a notebook...

#%%

sns.relplot(data=dfin, x="ds", y="demand", row="state", col="store", kind="line")


# dfp = dfin[dfin["ds"] > "2011-09-25"]

dfp = dfin.copy()  # dfin[dfin.store_id == 'TX_2']
# %%

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.lineplot(x="ds", y="demand", hue="store_id", data=dfp, ax=ax[0])
sns.lineplot(x="ds", y="sell_price", hue="store_id", data=dfp, ax=ax[1])
ax[0].set(title="FOODS_1_096 Weekly Sales and Price Changes")

# %%

dfp = dfp[dfp.demand != 0]
cutoff = "2014-06-01"
train = dfp[dfp.ds <= cutoff]
test = dfp[dfp.ds > cutoff]

#%%
"""
Pass the whole thing into prophet, assume beta is the same for every product

THIS did not work well because of the scale of the items

"""

# m = Prophet()
# m.add_regressor("xp")
# m.fit(train)

# forecast = m.predict(test)
# ypred = forecast["yhat"]

# fig = m.plot_components(forecast)
# fig2 = m.plot(forecast)

# regressor_coefficients(m)
# # ypred = np.exp(forecast["yhat"])
# ypred = forecast["yhat"]
# mean_absolute_percentage_error(test["demand"], ypred)

# sMAPE(test["demand"].values, ypred)

# fig, ax = plt.subplots(figsize=(7, 4.5))
# ax.plot(test["demand"], ypred, "bo")
# ax.set(xlabel="True Values", ylabel="Predicted Values")
# %%

"""
Loop through each store, fitting prophet independently

This worked better

"""

error = []

for store in dfin["store_id"].unique()[:8]:
    print(store)
    dfp = dfin[dfin.store_id == store]
    train = dfp[dfp.ds <= cutoff]
    test = dfp[dfp.ds > cutoff]

    m = Prophet()
    m.add_regressor("xp")
    m.fit(train)

    forecast = m.predict(test)
    ypred = forecast["yhat"]
    error.append(sMAPE(test["demand"].values, ypred))


# %%


# %%
