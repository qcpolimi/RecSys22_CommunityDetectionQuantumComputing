#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/2018

@author: Anonymous
"""

import traceback
import numpy as np

from recsys.Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from recsys.Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from recsys.Recommenders.NonPersonalizedRecommender import TopPop



from recsys.Evaluation.Evaluator import EvaluatorHoldout

def write_log_string(log_file, string):
    log_file.write(string)
    log_file.flush()


def run_dataset(dataset_class):

    from recsys.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
    from recsys.Data_manager.DataSplitter_Holdout import DataSplitter_Holdout
    from recsys.Data_manager.DataPostprocessing_K_Cores import DataPostprocessing_K_Cores
    from recsys.Data_manager.DataPostprocessing_Implicit_URM import DataPostprocessing_Implicit_URM

    try:

        write_log_string(log_file, "On dataset {}\n".format(dataset_class))

        dataset_reader = dataset_class()
        dataset_reader.load_data()
        dataset_loaded = dataset_reader.load_data()

        URM_all = dataset_loaded.get_URM_all()

        write_log_string(log_file, "DataReader URM min {:.2f}, URM max {:.2f}".format(np.min(URM_all.data), np.max(URM_all.data)))

        for ICM_name, ICM_object in dataset_loaded.get_loaded_ICM_dict().items():
            write_log_string(log_file, ", {} min {:.2f} - max {:.2f}".format(ICM_name, np.min(ICM_object.data), np.max(ICM_object.data)))


        write_log_string(log_file, "\nDataReader OK, ")


        dataset_reader = DataPostprocessing_K_Cores(dataset_reader, k_cores_value=2)
        dataset_reader = DataPostprocessing_Implicit_URM(dataset_reader)
        # dataReader_object = DataPostprocessing_User_sample(dataReader_object, user_quota=0.80)
        # dataReader_object = DataPostprocessing_User_min_interactions(dataReader_object, min_interactions=5)
        dataset_reader.load_data()
        dataset_loaded = dataset_reader.load_data()

        write_log_string(log_file, "DataPostprocessing OK, ")



        dataSplitter = DataSplitter_leave_k_out(dataset_reader, k_out_value=5)
        dataSplitter.load_data()
        dataSplitter.load_data()
        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

        write_log_string(log_file, "DataSplitter_leave_k_out OK, ")



        dataSplitter = DataSplitter_Holdout(dataset_reader)
        dataSplitter.load_data()
        dataSplitter.load_data()
        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

        write_log_string(log_file, "DataSplitter_Holdout OK, ")



        # dataSplitter = DataSplitter_k_fold_random_fromDataSplitter(dataset_reader, DataSplitter_Holdout)
        # dataSplitter.load_data()
        # dataSplitter.load_data()
        #
        # write_log_string(log_file, "DataSplitter_k_fold_random_fromDataSplitter-DataSplitter_Holdout OK, ")
        #
        # write_log_string(log_file, " PASS\n\n")

        #
        # dataSplitter = DataSplitter_ColdItems_k_fold(dataReader_object)
        #
        # dataSplitter.load_data()
        #
        # URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
        #
        #
        # log_file.write("On dataset {} PASS\n".format(dataset_class))
        # log_file.flush()




        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen=True)


        recommender = TopPop(URM_train)
        recommender.fit()
        _, results_run_string = evaluator.evaluateRecommender(recommender)

        log_file.write("On dataset {} - TopPop\n".format(dataset_class))
        log_file.write(results_run_string)
        log_file.flush()


        for ICM_name in dataSplitter.get_loaded_ICM_names():

            ICM_object = dataSplitter.get_ICM_from_name(ICM_name)

            recommender = ItemKNNCBFRecommender(ICM_object, URM_train)
            recommender.fit()
            _, results_run_string = evaluator.evaluateRecommender(recommender)

            log_file.write("On dataset {} - ICM {}\n".format(dataset_class, ICM_name))
            log_file.write(results_run_string)
            log_file.flush()



        for UCM_name in dataSplitter.get_loaded_UCM_names():

            UCM_object = dataSplitter.get_UCM_from_name(UCM_name)

            recommender = UserKNNCBFRecommender(UCM_object, URM_train)
            recommender.fit()
            _, results_run_string = evaluator.evaluateRecommender(recommender)

            log_file.write("On dataset {} - UCM {}\n".format(dataset_class, UCM_name))
            log_file.write(results_run_string)
            log_file.flush()

        write_log_string(log_file, " PASS\n\n")



    except Exception as e:

        print("On dataset {} Exception {}".format(dataset_class, str(e)))
        log_file.write("On dataset {} Exception {}\n\n".format(dataset_class, str(e)))
        log_file.flush()

        traceback.print_exc()


from recsys.Data_manager import *


if __name__ == '__main__':


    log_file_name = "./result_experiments/run_test_datasets.txt"

    dataset_class_list = [
        FrappeReader,
        FilmTrustReader,
        GowallaReader,


        Movielens100KReader,
        Movielens1MReader,
        Movielens10MReader,
        Movielens20MReader,
        TheMoviesDatasetReader,

        LastFMHetrec2011Reader,
        DeliciousHetrec2011Reader,
        MovielensHetrec2011Reader,

        BookCrossingReader,
        BrightkiteReader,
        EpinionsReader,
        NetflixPrizeReader,
        ThirtyMusicReader,
        YelpReader,


        SpotifyChallenge2018Reader,
        XingChallenge2016Reader,
        XingChallenge2017Reader,

        TVAudienceReader,
        CiteULike_aReader,
        CiteULike_tReader,

        AmazonAutomotiveReader,
        AmazonBooksReader,
        AmazonElectronicsReader,
        AmazonMoviesTVReader,

        # # MultifacetedMovieTrailerFeatureReader,
    ]

    log_file = open(log_file_name, "w")



    for dataset_class in dataset_class_list:
        run_dataset(dataset_class)
    #
    # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(run_dataset, dataset_list)

