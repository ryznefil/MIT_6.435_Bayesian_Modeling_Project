import torch

torch.zeros(10)

# from stan_models import Bayesian_model
# from utils import pickle_object
# import stan
# import time as t
#
# queried_item = "item_id == 'FOODS_2_058'"
#
# # model = Bayesian_model(day_start=1, day_end=1940)
# model = Bayesian_model(day_start=1, day_end=1520)
#
# model.query_model_data(query=queried_item)
#
# print("Starting First Model")
#
# # basic_results = model.fit_simple_homogeneous_model(
# #     num_samples=500, num_chains=4, quarterly_dummy=True, y_str="log_Q", x_str="log_P", train_date_split='2014-06-01')
# # pickle_object(basic_results, "model_outputs/log_basic_model.pkl")
# #
# # multiple_betas = model.fit_multiple_beta_homogeneous_model(
# #     num_samples=500, num_chains=4, quarterly_dummy=True, y_str="log_Q", x_str="log_P", train_date_split='2014-06-01')
# # pickle_object(multiple_betas, "model_outputs/log_multiple_betas.pkl")
#
# # basic_results = model.fit_simple_homogeneous_model(
# #     num_samples=500, num_chains=4, quarterly_dummy=True, y_str="Q", x_str="P", train_date_split='2014-06-01')
# # pickle_object(basic_results, "model_outputs/non_log_basic_model.pkl")
# #
# # multiple_betas = model.fit_multiple_beta_homogeneous_model(
# #     num_samples=500, num_chains=4, quarterly_dummy=True, y_str="Q", x_str="P", train_date_split='2014-06-01')
# # pickle_object(multiple_betas, "model_outputs/non_log_multiple_betas.pkl")
#
# multiple_betas = model.fit_heterogeneous_model_without_mixture(segmentation_id='state_id',num_samples=1000, num_chains=4, quarterly_dummy=True, y_str="Q", x_str="P", train_date_split='2014-06-01')
# pickle_object(multiple_betas, "model_outputs/secondjunk.pkl")
#
# multiple_betas = model.fit_heterogeneous_model_plotting(segmentation_id='state_id',num_samples=1000, num_chains=4, quarterly_dummy=True, y_str="Q", x_str="P", train_date_split='2014-06-01')
# pickle_object(multiple_betas, "model_outputs/firstjunk.pkl")
