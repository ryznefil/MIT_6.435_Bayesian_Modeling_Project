#%%

import pandas as pd

dfout = pd.read_csv("stan_examples/s2_ymet_xmet_hier_double.csv")

# %%

dfout.describe()
# %%


dfout["zbetamu.1.2"].hist()
# %%

dfout["zbetamu.2.2"].hist(bins=20)
# %%

dfout["zbeta.3.1"].hist()
# %%

dfout["zbeta.2.1"].hist()

# %%

dfout["zbeta.3.2"].hist()
# %%

dfout["zbeta.2.2"].hist()

# %%
