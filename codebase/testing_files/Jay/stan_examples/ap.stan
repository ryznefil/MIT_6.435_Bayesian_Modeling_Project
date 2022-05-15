data {
    int<lower=1> Nsubj ;
    int<lower=1> Ntotal ;
    vector[Ntotal] zy;  // seems like vector is an array of reals, use vectors instead..
    vector[Ntotal] zx;
    vector<lower=0>[Ntotal] zw ;
    vector<lower=1>[Ntotal] s;
  }



  parameters {
    matrix[Nsubj, 2] zbeta0; 
    matrix[Nsubj, 2] zbeta1; 
    matrix[Nsubj, 2] zbeta2; // these are all matrices now, len Nsubj
    real<lower=0> zsigma ;
    real<lower=0> nu ;
  }

  model {
    zsigma ~ uniform( 1.0E-3 , 1.0E+3 ) ;
    nu ~ exponential(1/30.0) ;

    for (r in 1:Nsubj) {
      if (w[r] == 1) {
          zbeta[r,1] ~ normal(0, 10);
          zbeta[r,2] ~ normal(0, 10);
      }
      
      else{
          zbeta[r,1] ~ normal(100, 2);
          zbeta[r,2] ~ normal(100, 2); 
      }

    }
  }  


data {
    int<lower=1> Nsubj ;
    int<lower=1> Ntotal ;
    int<lower=1> Ncols ; 
    vector[Ntotal] zy; 
    vector[Ntotal] zx;
    vector<lower=0>[Ntotal] zw ;
    array[Ntotal] int<lower=1> s; // must be ints, because its an index
    array[Nsubj] int<lower=1> g; // must be ints, because its an index
  }

  parameters {
    matrix[Nsubj, Ncols] zbeta; 
    real<lower=0> zsigma ;
    real<lower=0> nu ;
  }


  model {
    zsigma ~ uniform( 1.0E-3 , 1.0E+3 ) ;
    nu ~ exponential(1/30.0) ;

    for (r in 1:Nsubj) {
      if (g[r] == 1) {
          zbeta[r,1] ~ normal(0, 10);
          zbeta[r,2] ~ normal(0, 10);
      }
      
      else if (g[2]!=1) {
          zbeta[r,1] ~ normal(100, 2);
          zbeta[r,2] ~ normal(100, 2); 
      }

    }

    for ( i in 1:Ntotal ) {
      zy[i] ~ student_t( 
                nu ,
                zbeta[s[i],1] + zbeta[s[i],2] * zx[i], 
                zw[i]*zsigma ) ;
    }
  }  