#%% Imports
import numpy as np
import pandas as pd
import stan
from sklearn import datasets

#%%

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature

model_data = {
    "N": diabetes_y.shape[0],
    "y": diabetes_y,
    "x": diabetes_X[:, 2],
}


#%%

stan_code = """

data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
    for (n in 1:N) {
        y[n] ~ normal(alpha + beta * x[n], sigma);
    }
}

"""

#%%

posterior = stan.build(stan_code, data=model_data)
fit = posterior.sample(num_chains=4, num_samples=1000)
df = fit.to_frame().to_csv("s1_fit.csv")

