#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/06/2020

@author: Anonymous
"""

import numpy as np
import os, traceback, multiprocessing
from argparse import ArgumentParser
from functools import partial

from recsys.Evaluation.Evaluator import EvaluatorHoldout
from recsys.Recommenders.Recommender_import_list import *
from recsys.Data_manager import *

from recsys.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from recsys.HyperparameterTuning.SearchSingleCase import SearchSingleCase
from recsys.HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, runHyperparameterSearch_Content, runHyperparameterSearch_Hybrid

from recsys.Data_manager.data_consistency_check import assert_implicit_data, assert_disjoint_matrices

from recsys.Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics
from recsys.Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from recsys.Utils.all_dataset_stats_latex_table import all_dataset_stats_latex_table
from recsys.Data_manager.DataSplitter_Holdout import DataSplitter_Holdout
from recsys.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from recsys.Utils.RecommenderInstanceIterator import RecommenderInstanceIterator


from recsys.Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender

def _get_model_list_given_dataset(dataset_class, recommender_class_list, KNN_similarity_list, ICM_dict, UCM_dict):

    recommender_class_list = recommender_class_list.copy()

    # Model list format: recommender class, KNN heuristic, ICM/UCM name, ICM/UCM matrix
    model_list = []

    for recommender_class in recommender_class_list:

        if issubclass(recommender_class, BaseItemCBFRecommender):
            for ICM_name, ICM_object in ICM_dict.items():
                if recommender_class in [ItemKNNCBFRecommender, ItemKNN_CFCBF_Hybrid_Recommender]:
                    for KNN_similarity in KNN_similarity_list:
                        model_list.append((recommender_class, KNN_similarity, ICM_name, ICM_object))
                else:
                    model_list.append((recommender_class, None, ICM_name, ICM_object))

        elif issubclass(recommender_class, BaseUserCBFRecommender):
            for UCM_name, UCM_object in UCM_dict.items():
                if recommender_class in [UserKNNCBFRecommender, UserKNN_CFCBF_Hybrid_Recommender]:
                    for KNN_similarity in KNN_similarity_list:
                        model_list.append((recommender_class, KNN_similarity, UCM_name, UCM_object))
                else:
                    model_list.append((recommender_class, None, UCM_name, UCM_object))


        else:
            if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:
                for KNN_similarity in KNN_similarity_list:
                    model_list.append((recommender_class, KNN_similarity, None, None))

            else:
                model_list.append((recommender_class, None, None, None))

    # Removing cases that have an estimated training time of more than 30 days
    if dataset_class in [TheMoviesDatasetReader] and LightFMItemHybridRecommender in recommender_class_list:
        model_list.remove((LightFMItemHybridRecommender, None, "ICM_all", ICM_dict["ICM_all"]))

    if dataset_class in [GowallaReader] and SLIMElasticNetRecommender in recommender_class_list:
        model_list.remove((SLIMElasticNetRecommender, None, None, None))

    if dataset_class in [YelpReader] and LightFMCFRecommender in recommender_class_list:
        model_list.remove((LightFMCFRecommender, None, None, None))

    return model_list


def _optimize_single_model(model_tuple, URM_train, URM_train_last_test = None,
                          n_cases = None, n_random_starts = None, resume_from_saved = False,
                          save_model = "best", evaluate_on_test = "best", max_total_time = None,
                          evaluator_validation = None, evaluator_test = None, evaluator_validation_earlystopping = None,
                          metric_to_optimize = None, cutoff_to_optimize = None,
                          model_folder_path ="result_experiments/"):

    try:

        recommender_class, KNN_similarity, ICM_UCM_name, ICM_UCM_object = model_tuple

        if recommender_class in [ItemKNN_CFCBF_Hybrid_Recommender, UserKNN_CFCBF_Hybrid_Recommender,
                                 LightFMUserHybridRecommender, LightFMItemHybridRecommender]:
            runHyperparameterSearch_Hybrid(recommender_class,
                                           URM_train=URM_train,
                                           URM_train_last_test=URM_train_last_test,
                                           metric_to_optimize=metric_to_optimize,
                                           cutoff_to_optimize=cutoff_to_optimize,
                                           evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                           evaluator_validation=evaluator_validation,
                                           similarity_type_list=[KNN_similarity],
                                           evaluator_test=evaluator_test,
                                           max_total_time=max_total_time,
                                           output_folder_path=model_folder_path,
                                           parallelizeKNN=False,
                                           allow_weighting=True,
                                           save_model = save_model,
                                           evaluate_on_test = evaluate_on_test,
                                           resume_from_saved=resume_from_saved,
                                           ICM_name=ICM_UCM_name,
                                           ICM_object=ICM_UCM_object.copy(),
                                           n_cases=n_cases,
                                           n_random_starts=n_random_starts)

        elif issubclass(recommender_class, BaseItemCBFRecommender) or issubclass(recommender_class, BaseUserCBFRecommender):
            runHyperparameterSearch_Content(recommender_class,
                                            URM_train=URM_train,
                                            URM_train_last_test=URM_train_last_test,
                                            metric_to_optimize=metric_to_optimize,
                                            cutoff_to_optimize=cutoff_to_optimize,
                                            evaluator_validation=evaluator_validation,
                                            similarity_type_list=[KNN_similarity],
                                            evaluator_test=evaluator_test,
                                            output_folder_path=model_folder_path,
                                            parallelizeKNN=False,
                                            allow_weighting=True,
                                            save_model = save_model,
                                            evaluate_on_test = evaluate_on_test,
                                            max_total_time=max_total_time,
                                            resume_from_saved=resume_from_saved,
                                            ICM_name=ICM_UCM_name,
                                            ICM_object=ICM_UCM_object.copy(),
                                            n_cases=n_cases,
                                            n_random_starts=n_random_starts)

        else:

            runHyperparameterSearch_Collaborative(recommender_class, URM_train=URM_train,
                                           URM_train_last_test=URM_train_last_test,
                                           metric_to_optimize=metric_to_optimize,
                                           cutoff_to_optimize=cutoff_to_optimize,
                                           evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                           evaluator_validation=evaluator_validation,
                                           similarity_type_list = [KNN_similarity],
                                           evaluator_test=evaluator_test,
                                           max_total_time=max_total_time,
                                           output_folder_path=model_folder_path,
                                           resume_from_saved=resume_from_saved,
                                           parallelizeKNN=False,
                                           allow_weighting=True,
                                           save_model = save_model,
                                           evaluate_on_test = evaluate_on_test,
                                           n_cases=n_cases,
                                           n_random_starts=n_random_starts)


    except Exception as e:
        print("On CBF recommender {} Exception {}".format(model_tuple[0], str(e)))
        traceback.print_exc()







def _get_data_split_and_folders(dataset_class):

    dataset_reader = dataset_class()

    result_folder_path = "result_experiments/hyperopt_random_holdout_80_10_10/{}/".format(dataset_reader._get_dataset_name())
    # result_folder_path = "result_experiments/hyperopt_leave_1_out/{}/".format(dataset_reader._get_dataset_name())

    data_folder_path = result_folder_path + "data/"
    model_folder_path = result_folder_path + "models/"

    dataSplitter = DataSplitter_Holdout(dataset_reader, user_wise = False, split_interaction_quota_list=[80, 10, 10])
    # dataSplitter = DataSplitter_leave_k_out(dataset_reader, k_out_value = 1, use_validation_set = True, leave_random_out = True)
    dataSplitter.load_data(save_folder_path=data_folder_path)

    # Save statistics if they do not exist
    if not os.path.isfile(data_folder_path + "item_popularity_plot"):

        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

        plot_popularity_bias([URM_train + URM_validation, URM_test],
                             ["URM train", "URM test"],
                             data_folder_path + "item_popularity_plot")

        save_popularity_statistics([URM_train + URM_validation, URM_test],
                                   ["URM train", "URM test"],
                                   data_folder_path + "item_popularity_statistics.tex")

        all_dataset_stats_latex_table(URM_train + URM_validation + URM_test, dataset_reader._get_dataset_name(),
                                      data_folder_path + "dataset_stats.tex")

    return dataSplitter, result_folder_path, data_folder_path, model_folder_path






def read_data_split_and_search(dataset_class,
                               recommender_class_list,
                               KNN_similarity_to_report_list = [],
                               flag_baselines_tune=False,
                               flag_print_results=False,
                               metric_to_optimize = None,
                               cutoff_to_optimize = None,
                               cutoff_list = None,
                               n_cases = None,
                               max_total_time = None,
                               resume_from_saved = True,
                               n_processes = 4):

    dataSplitter, result_folder_path, data_folder_path, model_folder_path = _get_data_split_and_folders(dataset_class)

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    URM_train_last_test = URM_train + URM_validation

    # Ensure disjoint test-train split
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list)
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list = [cutoff_to_optimize])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)


    model_cases_list = _get_model_list_given_dataset(dataset_class, recommender_class_list, KNN_similarity_to_report_list,
                                                     dataSplitter.get_loaded_ICM_dict(),
                                                     dataSplitter.get_loaded_UCM_dict())


    _optimize_single_model_partial = partial(_optimize_single_model,
                                             URM_train=URM_train,
                                             URM_train_last_test=URM_train+URM_train_last_test,
                                             n_cases=n_cases,
                                             n_random_starts=int(n_cases/3),
                                             resume_from_saved=resume_from_saved,
                                             save_model="best",
                                             evaluate_on_test="best",
                                             evaluator_validation=evaluator_validation,
                                             evaluator_test=evaluator_test,
                                             max_total_time = max_total_time,
                                             evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                             metric_to_optimize=metric_to_optimize,
                                             cutoff_to_optimize=cutoff_to_optimize,
                                             model_folder_path=model_folder_path)


    if flag_baselines_tune:

        pool = multiprocessing.Pool(processes=n_processes, maxtasksperchild=1)
        resultList = pool.map(_optimize_single_model_partial, model_cases_list, chunksize=1)

        pool.close()
        pool.join()

        # for single_case in model_cases_list:
        #     try:
        #         _optimize_single_model_partial(single_case)
        #     except Exception as e:
        #         print("On recommender {} Exception {}".format(single_case[0], str(e)))
        #         traceback.print_exc()


    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)

        result_loader = ResultFolderLoader(model_folder_path,
                                           base_algorithm_list = None,
                                           other_algorithm_list = None,
                                           KNN_similarity_list = KNN_similarity_to_report_list,
                                           ICM_names_list = dataSplitter.get_loaded_ICM_dict().keys(),
                                           UCM_names_list = dataSplitter.get_loaded_UCM_dict().keys(),
                                           )

        result_loader.generate_latex_results(result_folder_path + "{}_latex_results.txt".format("accuracy_metrics"),
                                           metrics_list = ['RECALL', 'PRECISION', 'MAP', 'NDCG'],
                                           cutoffs_list = [cutoff_to_optimize],
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_results(result_folder_path + "{}_latex_results.txt".format("beyond_accuracy_metrics"),
                                           metrics_list = ["NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                           cutoffs_list = cutoff_list,
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_time_statistics(result_folder_path + "{}_latex_results.txt".format("time"),
                                           n_evaluation_users=n_test_users,
                                           table_title = None)




import run_hyperparameter_optimization_parallel_dataset as hyperopt_parallel_dataset
import mkl

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help='Baseline hyperparameter search', type=bool, default=True)
    parser.add_argument('-p', '--print_results',        help='Print results', type=bool, default=True)

    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky', 'euclidean']

    mkl.set_num_threads(4)
    # mkl.get_max_threads()


    dataset_list = [
        # AmazonMusicalInstrumentsReader,
        # BookCrossingReader,
        # BrightkiteReader,
        # CiaoReader,
        # CiteULike_aReader,
        # CiteULike_tReader,
        # ContentWiseImpressionsReader,
        # DeliciousHetrec2011Reader,
        # EpinionsReader,
        # FilmTrustReader,
        # FrappeReader,
        # GowallaReader,
        # JesterJokesReader,
        # LastFMHetrec2011Reader,
        # MillionSongDatasetTasteReader,
        # Movielens100KReader,
        # Movielens1MReader,
        # Movielens10MReader,
        # Movielens20MReader,
        # MovielensHetrec2011Reader,
        # NetflixPrizeReader,
        # PinterestReader,
        # TafengReader,
        # TheMoviesDatasetReader,
        # ThirtyMusicReader,
        # TVAudienceReader,
        # XingChallenge2016Reader,
        # XingChallenge2017Reader,
        # YelpReader,
    ]


    recommender_class_list = [
        Random,
        TopPop,
        GlobalEffects,
        SLIMElasticNetRecommender,
        UserKNNCFRecommender,
        IALSRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        # MatrixFactorization_AsySVD_Cython,
        EASE_R_Recommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        SLIM_BPR_Cython,
        RP3betaRecommender,
        PureSVDRecommender,
        NMFRecommender,
        UserKNNCBFRecommender,
        ItemKNNCBFRecommender,
        UserKNN_CFCBF_Hybrid_Recommender,
        ItemKNN_CFCBF_Hybrid_Recommender,
        LightFMCFRecommender,
        LightFMUserHybridRecommender,
        LightFMItemHybridRecommender,
        ]



    metric_to_optimize = 'NDCG'
    cutoff_to_optimize = 10
    cutoff_list = [5, 10, 20, 30, 40, 50, 100]
    max_total_time = 14*24*60*60  # 14 days
    n_cases = 50
    n_processes = 5

    hyperopt_parallel_dataset.read_data_split_and_search(dataset_list,
                                                recommender_class_list,
                                                KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                                flag_baselines_tune = input_flags.baseline_tune,
                                                metric_to_optimize = metric_to_optimize,
                                                cutoff_to_optimize = cutoff_to_optimize,
                                                cutoff_list = cutoff_list,
                                                n_cases = n_cases,
                                                max_total_time = max_total_time,
                                                n_processes = n_processes,
                                                resume_from_saved = True,
                                                )

    # If the search was done with parallel datasets then this sequential step will
    # only print the result tables
    for dataset_class in dataset_list:
        read_data_split_and_search(dataset_class,
                                   recommender_class_list,
                                   KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                   flag_baselines_tune = False,#input_flags.baseline_tune,
                                   flag_print_results = input_flags.print_results,
                                   metric_to_optimize = metric_to_optimize,
                                   cutoff_to_optimize = cutoff_to_optimize,
                                   cutoff_list = cutoff_list,
                                   n_cases = n_cases,
                                   max_total_time = max_total_time,
                                   n_processes = n_processes,
                                   resume_from_saved = True,
                                   )

