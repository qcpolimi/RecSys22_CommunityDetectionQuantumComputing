#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/06/2020

@author: Anonymous
"""

import multiprocessing
from functools import partial
from recsys.Evaluation.Evaluator import EvaluatorHoldout
from recsys.Data_manager.data_consistency_check import assert_disjoint_matrices
# from run_hyperparameter_optimization_parallel_model import _get_data_split_and_folders, _get_model_list_given_dataset, _optimize_single_model
import run_hyperparameter_optimization_parallel_model as hyperopt_parallel_model

def _optimize_single_dataset_model(dataset_model_tuple,
                          n_cases = None, n_random_starts = None, resume_from_saved = False,
                          save_model = "best", evaluate_on_test = "best", max_total_time = None,
                          metric_to_optimize = None, cutoff_list = None, cutoff_to_optimize = None):


    dataset_class = dataset_model_tuple[0]

    dataSplitter, result_folder_path, data_folder_path, model_folder_path = hyperopt_parallel_model._get_data_split_and_folders(dataset_class)

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    URM_train_last_test = URM_train + URM_validation

    # Ensure disjoint test-train split
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list)
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list = [cutoff_to_optimize])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)

    hyperopt_parallel_model._optimize_single_model(dataset_model_tuple[1:],
                           URM_train = URM_train,
                           URM_train_last_test = URM_train_last_test,
                           n_cases = n_cases,
                           n_random_starts = n_random_starts,
                           resume_from_saved = resume_from_saved,
                           save_model = save_model,
                           evaluate_on_test = evaluate_on_test,
                           max_total_time = max_total_time,
                           evaluator_validation = evaluator_validation,
                           evaluator_test = evaluator_test,
                           evaluator_validation_earlystopping = evaluator_validation_earlystopping,
                           metric_to_optimize = metric_to_optimize,
                           cutoff_to_optimize = cutoff_to_optimize,
                           model_folder_path = model_folder_path)




def _get_all_dataset_model_combinations_list(dataset_list, recommender_class_list, KNN_similarity_list):

    dataset_model_cases_list = []

    for dataset_class in dataset_list:

        dataSplitter, result_folder_path, data_folder_path, model_folder_path = hyperopt_parallel_model._get_data_split_and_folders(dataset_class)

        model_cases_list = hyperopt_parallel_model._get_model_list_given_dataset(dataset_class, recommender_class_list, KNN_similarity_list,
                                                         dataSplitter.get_loaded_ICM_dict(),
                                                         dataSplitter.get_loaded_UCM_dict())

        dataset_model_cases_list.extend([(dataset_class, *model_cases_tuple) for model_cases_tuple in model_cases_list])


    return dataset_model_cases_list





def read_data_split_and_search(dataset_list,
                               recommender_class_list,
                               KNN_similarity_to_report_list = [],
                               flag_baselines_tune=False,
                               metric_to_optimize = None,
                               cutoff_to_optimize = None,
                               cutoff_list = None,
                               n_cases = None,
                               max_total_time = None,
                               n_processes = 4,
                               resume_from_saved = True):

    dataset_model_cases_list = _get_all_dataset_model_combinations_list(dataset_list, recommender_class_list, KNN_similarity_to_report_list)


    _optimize_single_dataset_model_partial = partial(_optimize_single_dataset_model,
                                             n_cases=n_cases,
                                             n_random_starts=int(n_cases/3),
                                             resume_from_saved=resume_from_saved,
                                             save_model="best",
                                             evaluate_on_test="best",
                                             max_total_time = max_total_time,
                                             metric_to_optimize=metric_to_optimize,
                                             cutoff_to_optimize=cutoff_to_optimize,
                                             cutoff_list=cutoff_list)


    if flag_baselines_tune:

        pool = multiprocessing.Pool(processes=n_processes, maxtasksperchild=1)
        resultList = pool.map(_optimize_single_dataset_model_partial, dataset_model_cases_list, chunksize=1)

        pool.close()
        pool.join()

        # for single_case in dataset_model_cases_list:
        #     try:
        #         _optimize_single_dataset_model_partial(single_case)
        #     except Exception as e:
        #         print("On recommender {} Exception {}".format(single_case[0], str(e)))
        #         traceback.print_exc()
