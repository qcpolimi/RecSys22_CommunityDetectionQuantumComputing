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




def read_data_split_and_search(dataset_class,
                               flag_baselines_tune=False,
                               flag_print_results=False):

    dataset_reader = dataset_class()

    result_folder_path = "result_experiments/random_holdout_80_20/{}/".format(dataset_reader._get_dataset_name())
    data_folder_path = result_folder_path + "data/"
    model_folder_path = result_folder_path + "models/"

    dataSplitter = DataSplitter_Holdout(dataset_reader, user_wise = False, split_interaction_quota_list=[80, 10, 10])
    dataSplitter.load_data(save_folder_path=data_folder_path)

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    URM_train_last_test = URM_train + URM_validation


    # Ensure disjoint test-train split
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)


    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["URM train", "URM test"],
                         data_folder_path + "item_popularity_plot")

    save_popularity_statistics([URM_train + URM_validation, URM_test],
                               ["URM train", "URM test"],
                               data_folder_path + "item_popularity_statistics")
    #
    # all_dataset_stats_latex_table(URM_train + URM_validation + URM_test, dataset_class._get_dataset_name(),
    #                               data_folder_path + "dataset_stats.tex")



    collaborative_algorithm_list = [
        Random,
        TopPop,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        PureSVDRecommender,
        NMFRecommender,
        # IALSRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        # MatrixFactorization_AsySVD_Cython,
        EASE_R_Recommender,
        SLIM_BPR_Cython,
        SLIMElasticNetRecommender,
        ]

    parallel_collaborative_algorithm_list = [
        # MultiThreadSLIM_SLIMElasticNetRecommender
        ]

    # Removing algorithms having a training time which is too long
    if dataset_class in [BookCrossingReader, TheMoviesDatasetReader]:
        # Removing SLIM EN, training time exceeds:
        #  - 2 days on BookCrossingReader
        #  - 19 h on TheMoviesDatasetReader
        collaborative_algorithm_list.remove(SLIMElasticNetRecommender)


    metric_to_optimize = 'NDCG'
    cutoff_to_optimize = 10
    cutoff_list = [5, 10, 20, 30, 40, 50, 100]

    n_cases = 50
    n_random_starts = int(n_cases/3)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list)
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list = [cutoff_to_optimize])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)


    runHyperparameterSearch_Collaborative_partial = partial(runHyperparameterSearch_Collaborative,
                                                       URM_train=URM_train,
                                                       URM_train_last_test=URM_train_last_test,
                                                       metric_to_optimize=metric_to_optimize,
                                                       cutoff_to_optimize=cutoff_to_optimize,
                                                       evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                                       evaluator_validation=evaluator_validation,
                                                       similarity_type_list = KNN_similarity_to_report_list,
                                                       evaluator_test=evaluator_test,
                                                       output_folder_path=model_folder_path,
                                                       resume_from_saved=True,
                                                       parallelizeKNN=False,
                                                       allow_weighting=True,
                                                       n_cases=n_cases,
                                                       n_random_starts=n_random_starts)



    if flag_baselines_tune:

        pool = multiprocessing.Pool(processes=int(3), maxtasksperchild=1)
        resultList = pool.map(runHyperparameterSearch_Collaborative_partial, collaborative_algorithm_list)

        pool.close()
        pool.join()

        for recommender_class in parallel_collaborative_algorithm_list:
            try:
                runHyperparameterSearch_Collaborative_partial(recommender_class)
            except Exception as e:
                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()


        ###############################################################################################
        ##### Item Content Baselines

        for ICM_name, ICM_object in dataSplitter.get_loaded_ICM_dict().items():

            try:

                runHyperparameterSearch_Content(ItemKNNCBFRecommender,
                                                URM_train = URM_train,
                                                URM_train_last_test = URM_train + URM_validation,
                                                metric_to_optimize = metric_to_optimize,
                                                cutoff_to_optimize=cutoff_to_optimize,
                                                evaluator_validation = evaluator_validation,
                                                similarity_type_list = KNN_similarity_to_report_list,
                                                evaluator_test = evaluator_test,
                                                output_folder_path = model_folder_path,
                                                parallelizeKNN = True,
                                                allow_weighting = True,
                                                resume_from_saved = True,
                                                ICM_name = ICM_name,
                                                ICM_object = ICM_object.copy(),
                                                n_cases = n_cases,
                                                n_random_starts = n_random_starts)


                runHyperparameterSearch_Hybrid(ItemKNN_CFCBF_Hybrid_Recommender,
                                               URM_train = URM_train,
                                               URM_train_last_test = URM_train + URM_validation,
                                               metric_to_optimize = metric_to_optimize,
                                               cutoff_to_optimize=cutoff_to_optimize,
                                               evaluator_validation = evaluator_validation,
                                               similarity_type_list = KNN_similarity_to_report_list,
                                               evaluator_test = evaluator_test,
                                               output_folder_path = model_folder_path,
                                               parallelizeKNN = True,
                                               allow_weighting = True,
                                               resume_from_saved = True,
                                               ICM_name = ICM_name,
                                               ICM_object = ICM_object.copy(),
                                               n_cases = n_cases,
                                               n_random_starts = n_random_starts)

            except Exception as e:

                print("On CBF recommender for ICM {} Exception {}".format(ICM_name, str(e)))
                traceback.print_exc()




        ################################################################################################
        ###### User Content Baselines

        for UCM_name, UCM_object in dataSplitter.get_loaded_UCM_dict().items():

            try:

                runHyperparameterSearch_Content(UserKNNCBFRecommender,
                                                URM_train = URM_train,
                                                URM_train_last_test = URM_train + URM_validation,
                                                metric_to_optimize = metric_to_optimize,
                                                cutoff_to_optimize=cutoff_to_optimize,
                                                evaluator_validation = evaluator_validation,
                                                similarity_type_list = KNN_similarity_to_report_list,
                                                evaluator_test = evaluator_test,
                                                output_folder_path = model_folder_path,
                                                parallelizeKNN = True,
                                                allow_weighting = True,
                                                resume_from_saved = True,
                                                ICM_name = UCM_name,
                                                ICM_object = UCM_object.copy(),
                                                n_cases = n_cases,
                                                n_random_starts = n_random_starts)



                runHyperparameterSearch_Hybrid(UserKNN_CFCBF_Hybrid_Recommender,
                                               URM_train = URM_train,
                                               URM_train_last_test = URM_train + URM_validation,
                                               metric_to_optimize = metric_to_optimize,
                                               cutoff_to_optimize=cutoff_to_optimize,
                                               evaluator_validation = evaluator_validation,
                                               similarity_type_list = KNN_similarity_to_report_list,
                                               evaluator_test = evaluator_test,
                                               output_folder_path = model_folder_path,
                                               parallelizeKNN = True,
                                               allow_weighting = True,
                                               resume_from_saved = True,
                                               ICM_name = UCM_name,
                                               ICM_object = UCM_object.copy(),
                                               n_cases = n_cases,
                                               n_random_starts = n_random_starts)

            except Exception as e:

                print("On CBF recommender for UCM {} Exception {}".format(UCM_name, str(e)))
                traceback.print_exc()





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



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help='Baseline hyperparameter search', type=bool, default=True)
    parser.add_argument('-p', '--print_results',        help='Print results', type=bool, default=True)

    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky', 'euclidean']

    dataset_list = [
        # O AmazonMusicalInstrumentsReader,
        # O BookCrossingReader,
        # BrightkiteReader,
        # K CiteULike_aReader,
        # K CiteULike_tReader,
        # K DeliciousHetrec2011Reader,
        # EpinionsReader,
        # *FilmTrustReader,
        FrappeReader,
        # GowallaReader,
        # JesterJokesReader,
        # LastFMHetrec2011Reader,
        # Movielens100KReader,
        # Movielens1MReader,
        # Movielens10MReader,
        # Movielens20MReader,
        # MovielensHetrec2011Reader,
        # NetflixPrizeReader,
        # TheMoviesDatasetReader,
        # ThirtyMusicReader,
        # TVAudienceReader,
        # XingChallenge2016Reader,
        # XingChallenge2017Reader,
        # YelpReader,
    ]

    for dataset_class in dataset_list:
        read_data_split_and_search(dataset_class,
                                   flag_baselines_tune=input_flags.baseline_tune,
                                   flag_print_results=input_flags.print_results,
                                   )

