#%% imports

import pandas as pd
import stan

#%%

df = pd.read_csv("DBDA2Eprograms/HtWtData300.csv")
# %%

model_data = dict()
model_data["Ntotal"] = df.shape[0]
model_data["x"] = df.height.values
model_data["y"] = df.weight.values
# model_data["meanY"] = df.weight.mean()
# model_data["sdY"] = df.weight.std()
# model_data["meanX"] = df.height.mean()
# model_data["sdX"] = df.height.std()

print(model_data)


#%%
model_string = """

data {
    int<lower=1> Ntotal ;
    real x[Ntotal] ;
    real y[Ntotal] ;
  }
  transformed data {
    real unifLo ;
    real unifHi ;
    real expLambda ;
    real beta0sigma ;
    real beta1sigma ;
    real meanY ;
    real meanX ;
    real sdY ;
    real sdX ;

    meanY <- mean(y) ;
    meanX <- mean(x) ;
    sdY <- sd(y) ;
    sdX <- sd(x) ;
    unifLo <- sdY/1000 ;
    unifHi <- sdY*1000 ;
    expLambda <- 1/30.0 ;
    beta1sigma <- 10*fabs(sdY/sdX) ;
    beta0sigma <- 10*fabs(meanX*sdY/sdX) ;
  }
  parameters {
    real beta0 ;
    real beta1 ;
    real<lower=0> nu ; 
    real<lower=0> sigma ; 
  }
  model {
    sigma ~ uniform( unifLo , unifHi ) ; //variance of the data has priors from sdY
    nu ~ exponential( expLambda ) ; // nu has prior exponential(1/30.0)
    beta0 ~ normal( 0 , beta0sigma ) ; //prior 0, sigma close to sdY and sdX
    beta1 ~ normal( 0 , beta1sigma ) ;
    for ( i in 1:Ntotal ) {
      y[i] ~ student_t( nu , beta0 + beta1 * x[i] , sigma  ) ; // per model
    }
  }
"""

#%%

posterior = stan.build(program_code=model_string, data=model_data)
fit = posterior.sample(num_chains=4, num_samples=100)
print(fit.to_frame())

# %%
