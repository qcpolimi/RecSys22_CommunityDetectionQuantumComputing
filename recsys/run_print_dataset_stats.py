#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/10/19

@author: Anonymous
"""

from recsys.Data_manager import *
import traceback, os
import numpy as np
import scipy.sparse as sps
from recsys.Data_manager.DataReader_utils import compute_density


def gini_index(URM_all):

    URM_all = sps.csr_matrix(URM_all)
    array = np.ediff1d(URM_all.indptr)

    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = np.array(array, dtype=np.float64)
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


import pandas as pd


def val_stats(prefix, data_vector):

    result = {prefix + "val min": np.min(data_vector),
              prefix + "val max": np.max(data_vector),
              prefix + "val mean": np.mean(data_vector),
              prefix + "val std": np.std(data_vector),
              }

    return result



if __name__ == '__main__':


    result_file_folder = "./result_experiments/"

    log_file = open(result_file_folder + "dataset_stats.txt", "w")

    # If directory does not exist, create
    if not os.path.exists(result_file_folder):
        os.makedirs(result_file_folder)


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

        # NetflixEnhancedReader,

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
        AmazonMusicReader,

        # MultifacetedMovieTrailerFeatureReader,
    ]


    #
    # result_dataframe = pd.DataFrame(None, columns = ["Dataset name",
    #                                                  "URM name",
    #                                                  "n users", "n items", "n data points",
    #                                                  "ICM name",
    #                                                  "n_features",
    #                                                  "density", "sparsity",
    #                                                  "data val min", "data val max", "data val mean", "data val std",
    #                                                  "item profile len min", "item profile len max", "item profile len avg", "item profile len std",
    #                                                  "user profile len min", "user profile len max", "user profile len avg", "user profile len std",
    #                                                  "gini index item pop",
    #                                                  "gini index user profile",
    #                                                  "HAS ICM"]
    #                                 )

    result_dataframe = pd.DataFrame(None)

    for dataset_index, dataset_class in enumerate(dataset_class_list):

        try:

            dataset_object = dataset_class()
            dataset_object = dataset_object.load_data()

            log_file.write("[{}/{}] Dataset: {}\n".format(dataset_index+1, len(dataset_class_list),
                                                          dataset_object.get_dataset_name()))

            for URM_name, URM_object in dataset_object.get_loaded_URM_dict().items():

                log_file.write("\tURM {}: n_users {}, n_items {}, nnz {:.2E}, \tDensity {:.2E}, Gini {:.2f}\n".format(
                    URM_name, URM_object.shape[0], URM_object.shape[1], URM_object.nnz, compute_density(URM_object), gini_index(URM_object)))

                item_profile = np.ediff1d(sps.csc_matrix(URM_object).indptr)
                user_profile = np.ediff1d(sps.csr_matrix(URM_object).indptr)

                new_row = {"Dataset name":dataset_object.get_dataset_name(),
                         "URM name": URM_name,
                         "n users": URM_object.shape[0],
                         "n items": URM_object.shape[1],
                         "n data points": URM_object.nnz,
                         "density": compute_density(URM_object),
                         "sparsity": 1-compute_density(URM_object),
                         "gini index item pop": gini_index(URM_object),
                         "gini index user profile": gini_index(URM_object.T),
                         **val_stats("data ", URM_object.data),
                         **val_stats("item profile len ", item_profile),
                         **val_stats("user profile len ", user_profile),
                         "HAS ICM": len(dataset_object.get_loaded_ICM_dict().items())>0,
                        }

                result_dataframe = result_dataframe.append(new_row, ignore_index=True)

            for ICM_name, ICM_object in dataset_object.get_loaded_ICM_dict().items():

                log_file.write("\t\tICM {}: n_items {}, n_features {}, nnz {:.2E}\n".format(
                    ICM_name, ICM_object.shape[0], ICM_object.shape[1], ICM_object.nnz))

                new_row = {"Dataset name":dataset_object.get_dataset_name(),
                         "ICM name": ICM_name,
                         "n items": ICM_object.shape[0],
                         "n_features": ICM_object.shape[1],
                         "n data points": ICM_object.nnz,
                         "density": compute_density(ICM_object),
                         "sparsity": 1-compute_density(ICM_object),
                           **val_stats("data " ,ICM_object.data),
                        }

                result_dataframe = result_dataframe.append(new_row, ignore_index=True)


            log_file.write("\n")
            log_file.flush()

        except Exception as e:

            print("On dataset {} Exception {}".format(dataset_class, str(e)))
            traceback.print_exc()


        result_dataframe.to_csv(result_file_folder + "dataset_stats.csv", index=False)