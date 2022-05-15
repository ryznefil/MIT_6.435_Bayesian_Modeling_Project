import stan
import time as t
import numpy as np
import pickle as pkl
import pandas as pd


train_date_split='2014-06-01'
data = pd.read_csv('/Users/ryznerf/Documents/0_MIT/Spring_2022/0_Projects/1_Bayesian_Modelling_project/BayesianSales/main_code/test.csv')
data = data[:]

train_data =  data[data['date'] <= train_date_split]
test_data = data[data['date'] > train_date_split]
data = [train_data, test_data]
# data already has clusters in this case

cluster_level = 'store_id'
# cluster the data - training
for idx, ds in enumerate(data):
    store_labels = ds[cluster_level].unique()
    ticker_dic = {}
    for cluster, store in enumerate(store_labels):
        ticker_dic[store] = cluster + 1   # stan indexes from 1
    ds['cluster'] = ds[cluster_level].apply(lambda x: ticker_dic[x])
    data[idx] = ds


# global settings - training data
store_labels = data[0][cluster_level].unique()
K = len(store_labels) # number of segments
E = 3  # 3 different quarter dummies
N = len(data[0])


mu_segments = [0 for _ in range(K)]
target = ['log_Q']
group_regressors = ['log_P']
shared_regressors = ["q_" + str(i + 1) for i in range(1, 4)]

x = data[0][group_regressors].to_numpy().flatten()
y = data[0][target].to_numpy().flatten()
p = data[0][shared_regressors].to_numpy()
g = data[0]['cluster'].to_numpy() # observation segment membership

x_new = data[1][group_regressors].to_numpy().flatten()
p_new = data[1][shared_regressors].to_numpy()
g_new = data[1]['cluster'].to_numpy() # observation segment membership

N_new = len(x_new)

model_data = {
    'y': y,
    'x': x,
    'g': g,
    'K': K,
    'N': N,
    'E': E,
    'p': p,
    'mu_segment' : mu_segments,
    'N_new': N_new,
    'x_new': x_new,
    'g_new': g_new,
    'p_new': p_new,
}


## NOTE: Number of segment variables is not included as we only have one - price
model_code = """
            data {
              // TRAINING
              // data lenghts
              int<lower = 1> N;               // Number of sales observations
              int<lower = 1> K;               // Number of segments
              int<lower = 1> E;               // Number of shared predictors
            
              // observations
              vector[N] y;                    // Vector of targets
              vector[N] x;                    // Vector of variables
              matrix[N,E] p;                  // Matrix of shared predictors values
              int<lower = 1, upper = K> g[N]; // Vector of group assignments
              
              // prior conditioning
              vector[K] mu_segment;          // Mean of the segments model.
              
              // PREDICTIONS
              int<lower=1>N_new;            // Number of sales observations for prediction
              vector[N_new] x_new;                    // Vector of observed variables for prediction
              matrix[N_new,E] p_new;                  // Matrix of shared predictors values
              int<lower = 1, upper = K> g_new[N_new]; // Vector of group assignments for prediction
              
            }
            transformed data {
              cov_matrix[1] Q = [[1]];
            }
            
            // Parameters and hyperparameters.
            parameters {
              real<lower = 0> sigma;          // Variance of the regression
            
              vector[K] alpha;                // Vectors of intercepts per segment
              vector[K] beta;                 // Vector of segment coefficients
              vector[E] delta;                // Vector of shared predictors coefficients
              real<lower = 0> tau[K];         // Variance of the segments model
            }
            
            // Hierarchical regression.
            model {
              // Hyperpriors.
              tau ~ inv_gamma(1, 1);                // Hyper-prior on the variance of individual segments
            
              // Priors
              alpha ~ normal(0,10);                 // intercept priors
              sigma ~ inv_gamma(0.001, 0.001);      // error term prior
              delta ~ normal(0,10);                 // shared predictors coefficients
            
              // Drawing betas for clusters
              for (k in 1:K){
                beta[k] ~ normal(mu_segment[k], tau[k]);
              }
            
              for (n in 1:N) {
                y[n] ~ normal(alpha[g[n]] + p[n]*delta  + x[n]*beta[g[n]], sigma);
              }
            }
            generated quantities {
                vector[N_new] y_new;
                for (n_new in 1:N_new){
                    y_new[n_new] = normal_rng(alpha[g_new[n_new]] + p_new[n_new]*delta  + x_new[n_new]*beta[g[n_new]], sigma);
                }
            }
            """

start = t.time()
posterior = stan.build(model_code, data=model_data)
fit = posterior.sample(num_chains=4, num_samples=250)
fit = fit.to_frame()
end = t.time()
print("Sampling and prediction took: ", end - start)

