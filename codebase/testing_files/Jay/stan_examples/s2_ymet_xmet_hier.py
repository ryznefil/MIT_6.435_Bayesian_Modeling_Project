#%% imports

import numpy as np
import pandas as pd
import stan

#%%
df = pd.read_csv("DBDA2Eprograms/HierLinRegressData.csv")

df = df[df.Subj < 5]


df
# %%

model_data = {
    "Nsubj": df.Subj.max(),
    "Ntotal": df.shape[0],
    "y": df.Y.values,
    "x": df.X.values,
    "w": np.ones(df.shape[0]),
    "s": df.Subj.values,
}

print(model_data)


#%%
model_string = """
  data {
    int<lower=1> Nsubj ;
    int<lower=1> Ntotal ;
    array[Ntotal] real y;
    array[Ntotal] real x;
    array[Ntotal] real<lower=0> w ;
    array[Ntotal] int<lower=1> s;
  }
  transformed data { // this part is just for standardinzing the input data
    real zx[Ntotal] ;
    real zy[Ntotal] ;
    real zw[Ntotal] ;
    real wm ;
    real xm ;
    real ym ;
    real xsd ;
    real ysd ;
    xm = mean(x) ;
    ym = mean(y) ;
    wm = mean(w) ;
    xsd = sd(x) ;
    ysd = sd(y) ;
    for ( i in 1:Ntotal ) { // standardizing the data 
      zx[i] = ( x[i] - xm ) / xsd ; 
      zy[i] = ( y[i] - ym ) / ysd ; 
      zw[i] = w[i] / wm  ;
    }
  }

  parameters {
    real zbeta0[Nsubj] ; 
    real zbeta1[Nsubj] ;
    real zbeta2[Nsubj] ; // these are all vectors now, len Nsubj
    real<lower=0> zsigma ;
    real zbeta0mu ; 
    real zbeta1mu ; 
    real zbeta2mu ; 
    real<lower=0> zbeta0sigma ;
    real<lower=0> zbeta1sigma ;
    real<lower=0> zbeta2sigma ;
    real<lower=0> nu ;
  }

  transformed parameters {
    real beta0[Nsubj] ;  // apply a transformation to the actual betas...
    real beta1[Nsubj] ;
    real beta2[Nsubj] ;
    real<lower=0> sigma ;
    real beta0mu ; 
    real beta1mu ; 
    real beta2mu ; 
    
    for ( j in 1:Nsubj ) { 
      beta2[j] = zbeta2[j]*ysd/square(xsd) ; // scaling beta2[j] by the same amount
      beta1[j] = zbeta1[j]*ysd/xsd - 2*zbeta2[j]*xm*ysd/square(xsd) ; // confused
      beta0[j] = zbeta0[j]*ysd  + ym - zbeta1[j]*xm*ysd/xsd + zbeta2[j]*square(xm)*ysd/square(xsd) ;
    }
    beta2mu = zbeta2mu*ysd/square(xsd) ;
    beta1mu = zbeta1mu*ysd/xsd - 2*zbeta2mu*xm*ysd/square(xsd) ;
    beta0mu = zbeta0mu*ysd  + ym - zbeta1mu*xm*ysd/xsd + zbeta2mu*square(xm)*ysd/square(xsd) ;
    sigma = zsigma * ysd ;
  } 

  model {
    zbeta0mu ~ normal( 0 , 10 ) ;
    zbeta1mu ~ normal( 0 , 10 ) ;
    zbeta2mu ~ normal( 0 , 10 ) ;
    zsigma ~ uniform( 1.0E-3 , 1.0E+3 ) ;
    zbeta0sigma ~ uniform( 1.0E-3 , 1.0E+3 ) ;
    zbeta1sigma ~ uniform( 1.0E-3 , 1.0E+3 ) ;
    zbeta2sigma ~ uniform( 1.0E-3 , 1.0E+3 ) ;
    nu ~ exponential(1/30.0) ;
    zbeta0 ~ normal( zbeta0mu , zbeta0sigma ) ; // vectorized
    zbeta1 ~ normal( zbeta1mu , zbeta1sigma ) ; // vectorized
    zbeta2 ~ normal( zbeta2mu , zbeta2sigma ) ; // vectorized
    for ( i in 1:Ntotal ) {
      zy[i] ~ student_t( 
                nu ,
                zbeta0[s[i]] + zbeta1[s[i]] * zx[i] + zbeta2[s[i]] * square(zx[i]) , 
                zw[i]*zsigma ) ;
    }
  }  

"""

#%%

posterior = stan.build(program_code=model_string, data=model_data)
fit = posterior.sample(num_chains=4, num_samples=100)
print(fit.to_frame())

# %%
fit.to_frame().to_csv("stan_examples/s2_ymet_xmet_hier.csv")

# %%

import pandas as pd

dfout = pd.read_csv("stan_examples/s2_ymet_xmet_hier.csv")

print(df.columns)

#%%
ysd = df["Y"].std()
xsd = df["X"].std()

#%%
zcol = dfout["zbeta2.1"]

np.isclose(zcol * ysd / (xsd * xsd), dfout["beta2.1"].values)

#%%

# %%
