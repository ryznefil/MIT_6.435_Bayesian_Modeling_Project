#%% imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan

# https://mc-stan.org/docs/2_29/reference-manual/vector-and-matrix-data-types.html
# https://mc-stan.org/docs/2_18/stan-users-guide/multivariate-hierarchical-priors-section.html
#%%
df = pd.read_csv("DBDA2Eprograms/HierLinRegressData.csv")

df = df[df.Subj < 5]

df["X2"] = df["X"] ** 2

df.loc[df.Subj == 3, "Y"] = -df.loc[df.Subj == 3, "Y"]

#%% Plot it


plt.scatter(x=df["X"], y=df["Y"], c=df["Subj"])

## first frist make things vectors rather than arrays ?

## first make beta of 2 (adjustable via num inpurts) rather than zbeta1, zbeta2

## add new column "Group" for group 1 [1,2,3] vs group 2 [4,5]
## have group 1 betas have the same parent distribution, group 2 priors

## later re-introduce the priors such that
## hence we'll need a group 1 mu and group 2 mu
df
# %%

model_data = {
    "Nsubj": df.Subj.max(),
    "Ntotal": df.shape[0],
    "Ncols": 2,
    "Ngroups": 2,
    "zy": df.Y.values,
    "zx": df.X.values,
    "zw": np.ones(df.shape[0]),
    "s": df.Subj.values,
    "g": [1, 1, 2, 1],
}

print(model_data)


#%%
model_string = """
  data {
    int<lower=1> Nsubj ;
    int<lower=1> Ntotal ;
    int<lower=1> Ncols ; 
    int<lower=1> Ngroups;
    vector[Ntotal] zy; 
    vector[Ntotal] zx;
    vector<lower=0>[Ntotal] zw ;
    array[Ntotal] int<lower=1> s; // must be ints, because its an index
    array[Nsubj] int<lower=1> g; // must be ints, because its an index
  }

  parameters {
    matrix[Nsubj, Ncols] zbeta ;  
    matrix[Ngroups, Ncols] zbetamu ;
    real<lower=0> zsigma ;
    real<lower=0> nu ;
  }


  model {
    to_vector(zbetamu) ~ normal( 0, 10) ; //priors on group mus

    zsigma ~ uniform( 1.0E-3 , 1.0E+3 ) ;
    nu ~ exponential(1/30.0) ;

    for (r in 1:Nsubj) {
        zbeta[r,1] ~ normal(zbetamu[g[r],1], 20);
        zbeta[r,2] ~ normal(zbetamu[g[r],2], 20);
    }

    for ( i in 1:Ntotal ) {
      zy[i] ~ student_t( 
                nu ,
                zbeta[s[i],1] + zbeta[s[i],2] * zx[i], 
                zw[i]*zsigma ) ;
    }
  }  

"""

#%%

posterior = stan.build(program_code=model_string, data=model_data)
fit = posterior.sample(num_chains=4, num_samples=400)
print(fit.to_frame())

# %%
fit.to_frame().to_csv("stan_examples/s2_ymet_xmet_hier_double.csv")

# %%
