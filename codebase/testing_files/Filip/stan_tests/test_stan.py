import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import stan
import time as t

from processing import M5Data
import numpy

# load the raw data for the desired product
data_container = M5Data(day_start=1, day_end=1500)
data_container.make_subset(query="item_id == 'FOODS_3_090'")

# extra processing on the raw data
df_data = data_container.subset
df_data = df_data[['item_id', 'store_id', 'demand', 'date', 'sell_price']]

# log data
epsilon = 1e-10
df_data['log_Q'] = np.log(df_data['demand'] + epsilon)
df_data['log_P'] = np.log(df_data['sell_price'] + epsilon)

# create quarter seasonality dummy, create monthly seasonality dummy
df_data['date'] = pd.to_datetime(df_data['date'])
df_data['month'] = pd.DatetimeIndex(df_data['date']).month
df_data['quarter'] = pd.DatetimeIndex(df_data['date']).quarter
one_hot_quarter = pd.get_dummies(df_data['quarter'], prefix='q')
one_hot_month = pd.get_dummies(df_data['month'], prefix='m')
df_data = df_data.drop('quarter',axis = 1)
df_data = df_data.drop('month',axis = 1)
df_data = df_data.join(one_hot_quarter)
df_data = df_data.join(one_hot_month)

# plot the price to quantity data
# sns.scatterplot(data=df_data, x="log_P", y="log_Q")
# plt.show()

# Simple Bayesian regression with one Beta (quarterly seasonality)
regressors = ['log_P', 'q_2', 'q_3', 'q_4']
y = df_data['log_Q'].to_numpy()
x = df_data[regressors].to_numpy()

# center the data
# center_function = lambda x: x - x.mean()
# x = center_function(x)
# y = center_function(y)


model_data = {
    'y': y,
    'x': x,
    'N': x.shape[0],
    'K': x.shape[1]
}

model_code = """
   data {
       int<lower=0> N;
       int<lower=0> K;
       matrix[N,K] x;
       vector[N] y;
   }
   parameters {
       real alpha;
       vector[K] beta;
       real<lower=0> sigma;
   }
   model {
 
        
       y ~ normal(alpha + x*beta, sigma);
   }
   """
# alpha
# ~ normal(0, 10);
# beta
# ~ normal(0, 10);
# sigma
# ~ normal(0, 2.5);
start = t.time()
posterior = stan.build(model_code, data=model_data)
fit = posterior.sample(num_chains=4, num_samples=1000)
end = t.time()
df = fit.to_frame()
df.to_csv("test_fit_seasonality_included.csv")

print('Sampling took: ', end - start)







