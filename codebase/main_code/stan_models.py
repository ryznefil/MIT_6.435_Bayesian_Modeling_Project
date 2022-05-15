# %% Imports

import os
import sys

import pandas as pd
import statsmodels.api as sm

## Add data processing package
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(DATA_PATH)
print(sys.path)
from data_processing import M5Data

# %%
pd.options.mode.chained_assignment = None
import time as t

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
import stan
from scipy import stats
from sklearn.model_selection import train_test_split


# %%


class Bayesian_model:
    """
    Bayesian Model Class

    Parameters
    ---------
    day_start: Int, first day
    day_end: Int, last day
    
    """

    def __init__(self, day_start=800, day_end=1200):
        self.day_start = day_start
        self.day_end = day_end
        self.data_container = M5Data(day_start=day_start, day_end=day_end)
        self.last_query = None
        self.df_data_full = None
        self.last_fit = None

    def create_test_data(self, time_split, end = '2015-04-01'):
        data = self.df_data_full[self.df_data_full['date'] > time_split]
        data = data[data['date'] <= end]
        return data

    def create_train_data(self, time_split):
        return self.df_data_full[self.df_data_full['date'] <= time_split]

    def query_model_data(
            self,
            query="item_id == 'FOODS_3_090'",
            sampling_freq="W",
            group_by="store_id",
            epsilon=1e-30,
            train_perc=0.8,
            drop_nan = True
    ):
        """"Query and pre-process data for a specific item
        
        Parameters
        ----------
        query: string, applied to data container to find a subset
        sampling_freq: string, resample frequency
        group_by = string, aggregation level
        epsilon = float, amount to add to make sales non-zero
        train_perc = float, percentage of data to use for training
        
        """
        self.last_query = query
        self.data_container.make_subset(query=query)
        self.df_data_full = self.data_container.subset

        assert not self.df_data_full.empty

        self.df_data_full = self.df_data_full[
            ["item_id", "store_id", "state_id", "demand", "date", "sell_price"]
        ]

        # create quarter seasonality dummy, create monthly seasonality dummy
        self.df_data_full["date"] = pd.to_datetime(self.df_data_full["date"])

        # group data
        if group_by == "store_id":
            self.df_data_full = (
                self.df_data_full.groupby("store_id")
                    .resample(sampling_freq)
                    .agg(
                    {
                        "state_id": np.random.choice,
                        "demand": np.sum,
                        "sell_price": np.mean,
                    }
                )
                    .reset_index()
            )

        elif group_by == "state_id":
            self.df_data_full = (
                self.df_data_full.groupby("state_id")
                    .resample(sampling_freq)
                    .agg(
                    {
                        "store_id": np.random.choice,
                        "demand": np.sum,
                        "sell_price": np.mean,
                    }
                )
                    .reset_index()
            )

        # log price and demand
        ## maybe add an if statement for logs
        self.df_data_full["log_Q"] = np.log(self.df_data_full["demand"] + epsilon)
        self.df_data_full["log_P"] = np.log(self.df_data_full["sell_price"] + epsilon)
        self.df_data_full["P"] = self.df_data_full["sell_price"]
        self.df_data_full["Q"] = self.df_data_full["demand"]

        # create dummies
        # if month_dummy:
        self.df_data_full["month"] = pd.DatetimeIndex(self.df_data_full["date"]).month
        one_hot_month = pd.get_dummies(self.df_data_full["month"], prefix="m")
        self.df_data_full = self.df_data_full.drop("month", axis=1)
        self.df_data_full = self.df_data_full.join(one_hot_month)

        # if quarter_dummy:
        self.df_data_full["quarter"] = pd.DatetimeIndex(
            self.df_data_full["date"]
        ).quarter
        one_hot_quarter = pd.get_dummies(self.df_data_full["quarter"], prefix="q")
        self.df_data_full = self.df_data_full.drop("quarter", axis=1)
        self.df_data_full = self.df_data_full.join(one_hot_quarter)

        print("Processed data for query: " + query)
        self.df_data_full = self.df_data_full.dropna() # drop nan values

        return self.df_data_full

    def fit_simple_homogeneous_model(
            self,
            y_str="log_Q",
            x_str="log_P",
            quarterly_dummy=True,
            monthly_dummy=False,
            num_chains=4,
            num_samples=1000,
            train_date_split='2014-06-01'
    ):
        """ Fitting simple homgeneous model

        Args:
            y_str (str): Defaults to "log_Q".
            x_str (str): Defaults to "log_P".
            quarterly_dummy (bool, optional)
            monthly_dummy (bool, optional)
            num_chains (int, optional): Passed to Stan. Defaults to 4.
            num_samples (int, optional): Passed to Stan. Defaults to 1000.
            train_date_split (str, optional): Date to split train/test data

        Returns:
            results_container: fitted model samples from Stan
        """

        if self.df_data_full is None:
            raise ValueError("Data have not been prepared for the model")

        # define the set of regressors
        regressors = (
                [x_str]
                + quarterly_dummy * ["q_" + str(i + 1) for i in range(1, 4)]
                + monthly_dummy * ["m_" + str(i + 1) for i in range(1, 12)]
        )
        results_container = {}

        local_training_data = self.create_train_data(train_date_split)
        local_testing_data = self.create_test_data(train_date_split)

        # put data into ndarray for Stan
        y = local_training_data[y_str].to_numpy()
        x = local_training_data[regressors].to_numpy()

        x_reg = sm.add_constant(x, prepend=True)
        y_reg = y
        mod = sm.OLS(y_reg, x_reg)
        res = mod.fit()
        mu_alfa, mu_beta, mu_d2, mu_d3, mu_d4 = res.params

        x_new = local_testing_data[regressors].to_numpy()

        # dictionary with the data
        model_data = {"y": y,
                      "x": x,
                      "N": x.shape[0],
                      "K": x.shape[1],
                      'x_new': x_new,
                      "N_new": x_new.shape[0],
                      'mu_alfa' : mu_alfa,
                      'mu_beta': mu_beta,
                      'mu_d2' : mu_d2,
                      'mu_d3' : mu_d3,
                      'mu_d4': mu_d4,
                      }

        results_container["full"] = self._run_homogeneous_model_fit(model_data, num_chains, num_samples)
        self.last_fit = results_container

        return results_container

    def fit_multiple_beta_homogeneous_model(
            self,
            y_str="log_Q",
            x_str="log_P",
            segment_id="store_id",
            quarterly_dummy=True,
            monthly_dummy=False,
            num_chains=4,
            num_samples=1000,
            train_date_split='2014-06-01',
    ):
        """ Multiple Beta Homogenous model

        Args:
            y_str (str): Defaults to "log_Q".
            x_str (str): Defaults to "log_P".
            segment_id (str, optional): Used to make unique betas, via looping. Defaults to "store_id".
            quarterly_dummy (bool, optional)
            monthly_dummy (bool, optional)
            num_chains (int, optional): Passed to Stan. Defaults to 4.
            num_samples (int, optional): Passed to Stan. Defaults to 1000.
            train_date_split (str, optional): Date to split train/test data

        Returns:
            results_containter: fitted model samples from Stan
        """

        # define the regressors
        regressors = (
                [x_str]
                + quarterly_dummy * ["q_" + str(i + 1) for i in range(1, 4)]
                + monthly_dummy * ["m_" + str(i + 1) for i in range(1, 12)]
        )

        # get the unique segments of the data based omn the segmentation id
        segmentation_ids = self.df_data_full[segment_id].unique()

        # iterate over all the segments and fit model for them separately
        results_container = {}

        for segment in segmentation_ids:
            local_train = self.create_train_data(train_date_split)
            local_test = self.create_test_data(train_date_split)
            local_train = local_train[local_train[segment_id] == segment].reset_index(drop=True)
            local_test = local_test[local_test[segment_id] == segment].reset_index(drop=True)

            y = local_train[y_str].to_numpy()
            x = local_train[regressors].to_numpy()

            x_reg = sm.add_constant(x, prepend=True)
            y_reg = y
            mod = sm.OLS(y_reg, x_reg)
            res = mod.fit()
            mu_alfa, mu_beta, mu_d2, mu_d3, mu_d4 = res.params

            x_new = local_test[regressors].to_numpy()

            # dictionary with the data
            model_data = {"y": y,
                          "x": x,
                          "N": x.shape[0],
                          "K": x.shape[1],
                          'x_new': x_new,
                          "N_new": x_new.shape[0],
                          'mu_alfa': mu_alfa,
                          'mu_beta': mu_beta,
                          'mu_d2': mu_d2,
                          'mu_d3': mu_d3,
                          'mu_d4': mu_d4,
                          }

            results_container[segment] = self._run_homogeneous_model_fit(
                model_data, num_chains, num_samples
            )

        self.last_fit = results_container
        return results_container

    def fit_heterogeneous_model_without_mixture(
            self,
            y_str="log_Q",
            x_str="log_P",
            segmentation_id="store_id",
            quarterly_dummy=True,
            monthly_dummy=False,
            num_chains=4,
            num_samples=1000,
            train_date_split='2014-06-01',
    ):
        results_container = {}
        # split the data
        train_data = self.create_train_data(train_date_split)
        test_data = self.create_test_data(train_date_split)
        data_container = [train_data, test_data]

        # cluster the data
        cluster_level = segmentation_id

        # cluster the data
        for idx, ds in enumerate(data_container):
            store_labels = ds[cluster_level].unique()
            ticker_dic = {}
            for cluster, store in enumerate(store_labels):
                ticker_dic[store] = cluster + 1  # stan indexes from 1
            ds['cluster'] = ds[cluster_level].apply(lambda x: ticker_dic[x])
            data_container[idx] = ds

        # global settings - training data
        store_labels = data_container[0][cluster_level].unique()
        K = len(store_labels)  # number of segments
        E = 3  # 3 different quarter dummies
        N = len(data_container[0])

        mu_segments = [0 for _ in range(K)]
        target = [y_str]
        group_regressors = [x_str]
        shared_regressors = quarterly_dummy * ["q_" + str(i + 1) for i in range(1, 4)] + monthly_dummy * [
            "m_" + str(i + 1) for i in range(1, 12)]

        x = data_container[0][group_regressors].to_numpy().flatten()
        y = data_container[0][target].to_numpy().flatten()
        p = data_container[0][shared_regressors].to_numpy()
        g = data_container[0]['cluster'].to_numpy()  # observation segment membership

        x_new = data_container[1][group_regressors].to_numpy().flatten()
        p_new = data_container[1][shared_regressors].to_numpy()
        g_new = data_container[1]['cluster'].to_numpy()  # observation segment membership

        N_new = len(x_new)

        model_data = {
            'y': y,
            'x': x,
            'g': g,
            'K': K,
            'N': N,
            'E': E,
            'p': p,
            'mu_segment': mu_segments,
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
                      sigma ~ inv_gamma(1, 1);      // error term prior
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
        fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)
        end = t.time()
        print("Sampling and prediction took: ", end - start)
        results_container["full"] = fit.to_frame()

        return results_container


    # def fit_heterogeneous_model_without_mixture(
    #         self,
    #         y_str="log_Q",
    #         x_str="log_P",
    #         segmentation_id="store_id",
    #         quarterly_dummy=True,
    #         monthly_dummy=False,
    #         num_chains=4,
    #         num_samples=1000,
    #         train_date_split='2014-06-01',
    # ):
    #     results_container = {}
    #     # split the data
    #     train_data = self.create_train_data(train_date_split)
    #     test_data = self.create_test_data(train_date_split)
    #     data_container = [train_data, test_data]
    #
    #     # cluster the data
    #     cluster_level = segmentation_id
    #
    #     # cluster the data
    #     for idx, ds in enumerate(data_container):
    #         store_labels = ds[cluster_level].unique()
    #         ticker_dic = {}
    #         for cluster, store in enumerate(store_labels):
    #             ticker_dic[store] = cluster + 1  # stan indexes from 1
    #         ds['cluster'] = ds[cluster_level].apply(lambda x: ticker_dic[x])
    #         data_container[idx] = ds
    #
    #     # global settings - training data
    #     store_labels = data_container[0][cluster_level].unique()
    #     K = len(store_labels)  # number of segments
    #     E = 3  # 3 different quarter dummies
    #     N = len(data_container[0])
    #
    #     mu_segments = [0 for _ in range(K)]
    #     target = [y_str]
    #     group_regressors = [x_str]
    #     shared_regressors = quarterly_dummy*["q_" + str(i + 1) for i in range(1, 4)] + monthly_dummy * ["m_" + str(i + 1) for i in range(1, 12)]
    #     full_regressors = [x_str] + quarterly_dummy * ["q_" + str(i + 1) for i in range(1, 4)] + monthly_dummy * [
    #         "m_" + str(i + 1) for i in range(1, 12)]
    #     arr_mu_alfa = []
    #     arr_mu_beta = []
    #     arr_mu_d = []
    #
    #     for seg in range(1, K+1):
    #         train_dat = data_container[0]
    #         limited_tr = train_dat[train_dat['cluster'] == seg]
    #
    #     x = data_container[0][group_regressors].to_numpy().flatten()
    #     y = data_container[0][target].to_numpy().flatten()
    #     p = data_container[0][shared_regressors].to_numpy()
    #     g = data_container[0]['cluster'].to_numpy()  # observation segment membership
    #
    #     x_new = data_container[1][group_regressors].to_numpy().flatten()
    #     p_new = data_container[1][shared_regressors].to_numpy()
    #     g_new = data_container[1]['cluster'].to_numpy()  # observation segment membership
    #
    #     N_new = len(x_new)
    #
    #     model_data = {
    #         'y': y,
    #         'x': x,
    #         'g': g,
    #         'K': K,
    #         'N': N,
    #         'E': E,
    #         'p': p,
    #         'N_new': N_new,
    #         'x_new': x_new,
    #         'g_new': g_new,
    #         'p_new': p_new,
    #     }
    #
    #     ## NOTE: Number of segment variables is not included as we only have one - price
    #     model_code = """
    #                 data {
    #                   // TRAINING
    #                   // data lenghts
    #                   int<lower = 1> N;               // Number of sales observations
    #                   int<lower = 1> K;               // Number of segments
    #                   int<lower = 1> E;               // Number of shared predictors
    #
    #
    #                   // observations
    #                   vector[N] y;                    // Vector of targets
    #                   vector[N] x;                    // Vector of variables
    #                   matrix[N,E] p;                  // Matrix of shared predictors values
    #                   int<lower = 1, upper = K> g[N]; // Vector of group assignments
    #
    #
    #                   // PREDICTIONS
    #                   int<lower=1>N_new;            // Number of sales observations for prediction
    #                   vector[N_new] x_new;                    // Vector of observed variables for prediction
    #                   matrix[N_new,E] p_new;                  // Matrix of shared predictors values
    #                   int<lower = 1, upper = K> g_new[N_new]; // Vector of group assignments for prediction
    #
    #                 }
    #
    #                 // Parameters and hyperparameters.
    #                 parameters {
    #                   real<lower = 0> sigma;          // Variance of the regression
    #
    #                   vector[K] alpha;                // Vectors of intercepts per segment
    #                   vector[K] beta;                 // Vector of segment coefficients
    #                   vector[E] delta;                // Vector of shared predictors coefficients
    #                   real<lower = 0> tau[K];         // Variance of the segments model
    #                   vector[K] beta_mu_sample;
    #                 }
    #
    #                 // Hierarchical regression.
    #                 model {
    #                   // Hyperpriors.
    #                   tau ~ inv_gamma(1, 1);                // Hyper-prior on the variance of individual segments
    #
    #
    #                   // Priors
    #                   for (k in 1:K){
    #                     alpha[k] ~ normal(0 , 50);                 // intercept priors
    #                   }
    #
    #                   // Drawing betas for clusters
    #                   for (k in 1:K){
    #                     beta[k] ~ normal(0, tau[k]);
    #                   }
    #
    #                   for (k in 1:K){
    #                     delta[k] ~ normal(0,50);                 // shared predictors coefficients
    #                   }
    #
    #
    #                   sigma ~ inv_gamma(1, 1);      // error term prior
    #
    #                   for (n in 1:N) {
    #                     y[n] ~ normal(alpha[g[n]] + p[n]*delta  + x[n]*beta[g[n]], sigma);
    #                   }
    #                 }
    #                 generated quantities {
    #                     vector[N_new] y_new;
    #                     for (n_new in 1:N_new){
    #                         y_new[n_new] = normal_rng(alpha[g_new[n_new]] + p_new[n_new]*delta  + x_new[n_new]*beta[g[n_new]], sigma);
    #                     }
    #                 }
    #                 """
    #
    #     start = t.time()
    #     posterior = stan.build(model_code, data=model_data)
    #     fit = posterior.sample(num_chains= num_chains, num_samples = num_samples)
    #     end = t.time()
    #     print("Sampling and prediction took: ", end - start)
    #     results_container["full"] = fit.to_frame()
    #
    #     return results_container

    def fit_heterogeneous_model_plotting(
            self,
            y_str="log_Q",
            x_str="log_P",
            segmentation_id="store_id",
            quarterly_dummy=True,
            monthly_dummy=False,
            num_chains=4,
            num_samples=1000,
            train_date_split='2014-06-01',
    ):
        results_container = {}
        # split the data
        full_data = self.df_data_full
        train_data = self.create_train_data(train_date_split)
        test_data = self.create_test_data(train_date_split)
        data_container = [train_data, test_data]

        # cluster the data
        cluster_level = segmentation_id

        # cluster the data
        for idx, ds in enumerate(data_container):
            store_labels = ds[cluster_level].unique()
            ticker_dic = {}
            for cluster, store in enumerate(store_labels):
                ticker_dic[store] = cluster + 1  # stan indexes from 1
            ds['cluster'] = ds[cluster_level].apply(lambda x: ticker_dic[x])
            data_container[idx] = ds

        store_labels = full_data[cluster_level].unique()
        ticker_dic = {}
        for cluster, store in enumerate(store_labels):
            ticker_dic[store] = cluster + 1  # stan indexes from 1
        full_data['cluster'] = full_data[cluster_level].apply(lambda x: ticker_dic[x])

        # global settings - training data
        store_labels = data_container[0][cluster_level].unique()
        K = len(store_labels)  # number of segments
        E = 3  # 3 different quarter dummies
        N = len(data_container[0])

        mu_segments = [0 for _ in range(K)]
        target = [y_str]
        group_regressors = [x_str]
        shared_regressors = quarterly_dummy * ["q_" + str(i + 1) for i in range(1, 4)] + monthly_dummy * [
            "m_" + str(i + 1) for i in range(1, 12)]

        x = data_container[0][group_regressors].to_numpy().flatten()
        y = data_container[0][target].to_numpy().flatten()
        p = data_container[0][shared_regressors].to_numpy()
        g = data_container[0]['cluster'].to_numpy()  # observation segment membership

        x_new = x
        p_new = p
        g_new = g

        # x_new = full_data[group_regressors].to_numpy().flatten()
        # p_new = full_data[shared_regressors].to_numpy()
        # g_new = full_data['cluster'].to_numpy()  # observation segment membership

        N_new = len(x_new)

        model_data = {
            'y': y,
            'x': x,
            'g': g,
            'K': K,
            'N': N,
            'E': E,
            'p': p,
            'mu_segment': mu_segments,
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
                       sigma ~ inv_gamma(1, 1);      // error term prior
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
        fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)
        end = t.time()
        print("Sampling and prediction took: ", end - start)
        results_container["full"] = fit.to_frame()

        return results_container


    @staticmethod
    def _run_homogeneous_model_fit(model_data, num_chains, num_samples):
        """
        Given the data, fit the model

        Parameters:
        -----------
        model_data: dictionary containing keys N, K, x and y
        num_chains: Int
        num_samples: Int, length of chains
        
        Returns:
        --------
        Fitted samples, as a DataFrame
        
        """

        model_code = """
                   data {
                       int<lower=0> N;       // Number of observations
                       int<lower=0> N_new;  // Number of predictions
                       int<lower=0> K;      // Number of variables
                       matrix[N,K] x;       // Variable values matrix
                       matrix[N_new,K] x_new;   // Prediction inputs matrix
                       vector[N] Q;         // Observed values of Q
                       
                       real mu_alfa;       
                       real mu_beta;
                       real mu_d2;
                       real mu_d3;
                       real mu_d4;
                   }
                   parameters {
                       real<lower=0>tau;        
                       real alpha;          
                       vector[K] beta;
                       real<lower=0> sigma;
                   }
                   model {
                       tau ~ inv_gamma(1, 1);       // sampling hyperparameter    
                       alpha ~ normal(mu_alfa, 50);     // sampling intercept
                       beta[1] ~ normal(mu_beta, tau);  // sampling price Beta
                       beta[2] ~ normal(mu_d2, 50);     // sampling second quarted coefficient
                       beta[3] ~ normal(mu_d3, 50);
                       beta[4] ~ normal(mu_d4, 50);
                       sigma ~ inv_gamma(0.001, 0.001);        // sampling sigma for error term

                       y ~ normal(alpha + x*beta, sigma);       // fitting
                   }
                    generated quantities {      // prediction
                        vector[N_new] y_new;
                        for (n in 1:N_new)
                        y_new[n] = normal_rng(x_new[n] * beta, sigma);
                    }
                   """
        start = t.time()
        posterior = stan.build(model_code, data=model_data)
        fit = posterior.sample(num_chains=num_chains, num_samples=num_samples)
        end = t.time()
        print("Sampling and prediction took: ", end - start)

        return fit.to_frame()


# simple model plotting
def plot_simple_model_results(samples_df, shop_name, save=False, save_location=None, h=1, w=5, logged=True
                              ):
    """Plots Beta values given fitted samples 

    Args:
        samples_df (dataframe): Samples from stan model fit
        shop_name (string): Used in title of plot
        save (bool, optional): Whether to save plot. Defaults to False.
        save_location (_type_, optional): _description_. Defaults to None.
        h (int, optional): Height of plot. Defaults to 1.
        w (int, optional): With of plot. Defaults to 5.
        logged (bool, optional): If beta was in log space. Defaults to True.

    Raises:
        RuntimeError: _description_
    """
    if save and save_location is None:
        raise RuntimeError("Location and name needs to be given to save the plot")

    fig, axes = plt.subplots(h, w, figsize=(30, 5), sharey=True)
    fig.suptitle("Simple linear model for store/stores: " + shop_name)

    # Intercept
    sns.histplot(ax=axes[0], data=samples_df, x="alpha", kde=True)
    axes[0].axvline(samples_df["alpha"].median(), color="r")
    axes[0].set_title("Alpha")

    # Beta
    if logged:
        sns.histplot(ax=axes[1], data=samples_df, x="beta.1", kde=True)
        axes[1].axvline(samples_df["beta.1"].median(), color="r")
        axes[1].set_title("Beta_log_price")
    else:
        sns.histplot(ax=axes[1], data=samples_df, x="beta.1", kde=True)
        axes[1].axvline(samples_df["beta.1"].median(), color="r")
        axes[1].set_title("Beta_price")

    # Beta_q2
    sns.histplot(ax=axes[2], data=samples_df, x="beta.2", kde=True)
    axes[2].axvline(samples_df["beta.2"].median(), color="r")
    axes[2].set_title("Beta_q2")

    # Beta_q3
    sns.histplot(ax=axes[3], data=samples_df, x="beta.3", kde=True)
    axes[3].axvline(samples_df["beta.3"].median(), color="r")
    axes[3].set_title("Beta_q3")

    # Beta_q4
    g = sns.histplot(ax=axes[4], data=samples_df, x="beta.4", kde=True)
    axes[4].axvline(samples_df["beta.4"].median(), color="r")
    axes[4].set_title("Beta_q4")

    if save:
        plt.savefig(save_location)
    plt.show()


def sMAPE(actual, forecast):
    multiplier = 100 / len(actual)
    nominator = np.array(2 * np.abs(forecast - actual)).astype(float)
    denominator = np.array(np.abs(actual) + np.abs(forecast)).astype(float)

    return multiplier * np.sum(
        np.divide(
            nominator, denominator, out=np.zeros_like(nominator), where=denominator != 0
        )
    )

def make_prediction_stan_sampling(df_sampling, df_test, logged):
    prediction_len = len(df_test)
    y_hat = np.mean(df_sampling[['y_new.' + str(i) for i in range(1, prediction_len + 1)]],axis = 0).to_numpy()

    if logged:
        actual_y = df_test["log_Q"]
    else:
        actual_y = df_test["demand"]

    sMAPE_value = sMAPE(actual=actual_y, forecast=y_hat)

    return y_hat, sMAPE_value

