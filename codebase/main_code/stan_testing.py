import numpy as np

from stan_models import Bayesian_model
from utils import pickle_object, unpickle_object
import stan
import time as t
from stan_models import make_prediction_stan_sampling

train_date_split='2014-06-01'

kept_foods = [
    'FOODS_1_093',
'FOODS_1_096',
'FOODS_1_199',
'FOODS_2_058',
'FOODS_2_298',]

kept_hobbies = [
    'HOBBIES_1_157',
'HOBBIES_1_312',
'HOBBIES_1_381',
'HOBBIES_2_048',
'HOBBIES_2_144',]

kept_households = [
    'HOUSEHOLD_2_155',
    'HOUSEHOLD_2_159',
    'HOUSEHOLD_2_235',
    'HOUSEHOLD_2_296',
    'HOUSEHOLD_2_376',]

testing_labels = np.concatenate((kept_foods, kept_hobbies))
testing_labels = np.concatenate((testing_labels, kept_households))

skipped_labels = []
failed_labels = []

cluster_region_results = {}
cluster_results = {}
multiple_results = {}
simple_results = {}

cluster_region_mape = {}
cluster_mape = {}
multiple_mape = {}
simple_mape = {}

run_simple = True
run_multiple = True
run_cluster = True
run_cluster_region = True

model = Bayesian_model(day_start=1, day_end=1520)
num_smp = 10

for item_label in testing_labels:
    print("New iteration")
    queried_item = "item_id == " + "'" + item_label + "'"
    model.query_model_data(query=queried_item)

    print("Starting Modelling for: ", queried_item)

    try:
        if run_simple:
            simple_beta = model.fit_simple_homogeneous_model(
                num_samples=num_smp, num_chains=4, quarterly_dummy=True, y_str="Q", x_str="P",
                train_date_split=train_date_split)
            simple_results[item_label] = simple_beta
            pickle_object(simple_results, "simple_results0.pkl")

        if run_multiple:
            multiple_betas = model.fit_multiple_beta_homogeneous_model(
                num_samples=num_smp, num_chains=4, quarterly_dummy=True, y_str="Q", x_str="P", train_date_split=train_date_split)
            multiple_results[item_label] = multiple_betas
            pickle_object(multiple_results, "multi_beta_results0.pkl")

        if run_cluster:
            cluster_beta = model.fit_heterogeneous_model_without_mixture(segmentation_id='store_id', num_samples=num_smp,
                                                                           num_chains=4, quarterly_dummy=True, y_str="Q",
                                                                            x_str="P", train_date_split='2014-06-01')
            cluster_results[item_label] = cluster_beta
            pickle_object(cluster_results, "cluster_results0.pkl")

        if run_cluster_region:
            cluster_region_beta = model.fit_heterogeneous_model_without_mixture(segmentation_id='state_id', num_samples=num_smp,
                                                                           num_chains=4, quarterly_dummy=True, y_str="Q",
                                                                           x_str="P", train_date_split='2014-06-01')
            cluster_region_results[item_label] = cluster_region_beta
            pickle_object(cluster_region_results, "cluster_results_region0.pkl")
    except:
        print('Failed training for: ', item_label)
        failed_labels.append(item_label)

    try:
        # simple
        if run_simple:
            df_simple = simple_beta['full']
            local_test = model.create_test_data(train_date_split)
            _, smape = make_prediction_stan_sampling(df_simple, local_test, False)
            simple_mape[item_label] = smape
            pickle_object(simple_mape, "simple_mape0.pkl")

        # homogeneous
        if run_multiple:
            store_idx = list(multiple_betas.keys())
            smape_cont = []
            for store in store_idx:
                store_df = multiple_betas[store]
                local_test = model.create_test_data(train_date_split)
                local_test = local_test[local_test['store_id'] == store].reset_index(drop=True)
                _, smape = make_prediction_stan_sampling(store_df, local_test, False)
                smape_cont.append(smape)
            multiple_mape[item_label] = smape_cont
            pickle_object(multiple_mape, "multiple_mape0.pkl")

        # clustering
        if run_cluster:
            df_simple = cluster_beta['full']
            local_test = model.create_test_data(train_date_split)
            _, smape = make_prediction_stan_sampling(df_simple, local_test, False)
            cluster_mape[item_label] = smape
            pickle_object(cluster_mape, "cluster_mape0.pkl")

        if run_cluster_region:
            df_simple = cluster_region_beta['full']
            local_test = model.create_test_data(train_date_split)
            _, smape = make_prediction_stan_sampling(df_simple, local_test, False)
            cluster_region_mape[item_label] = smape
            pickle_object(cluster_region_mape, "cluster_region_mape0.pkl")

    except:
        print('Failed evaluation for: ', item_label)
        failed_labels.append(item_label)

    print("iteration end")


