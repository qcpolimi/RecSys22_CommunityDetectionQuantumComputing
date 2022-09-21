#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/01/18

@author: Anonymous
"""


import zipfile, os, shutil, h5py
import scipy.io
import pandas as pd
import scipy.sparse as sps
from recsys.Data_manager.DataReader import DataReader
from recsys.Data_manager.DatasetMapperManager import DatasetMapperManager



class _CiteULikeReader(DataReader):
    """
    Datasets from:
    Xiaopeng Li and James She. 2017. Collaborative variational autoencoder for recommender systems.
    In Proceedings ofthe 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 305–314
    """

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EcjHpkI8TQdHnFVwVMkNGN4BmNkurMWw79sU8kpt4wk8eA?e=QYhdbz"
    AVAILABLE_ICM = ["ICM_title_abstract"]
    DATASET_SPECIFIC_MAPPER = []

    IS_IMPLICIT = True



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self.zip_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + "CiteULike/"
        self.decompressed_zip_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            self.dataFile = zipfile.ZipFile(self.zip_file_folder + "CiteULike_a_t.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file.")
            self._print("Automatic download not available, please ensure the ZIP data file is in folder {}.".format(self.zip_file_folder))
            self._print("Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(self.zip_file_folder):
                os.makedirs(self.zip_file_folder)

            raise FileNotFoundError("Automatic download not available.")


        local_dataset_name = "citeulike-{}".format(self.dataset_variant)
        train_interactions_file_suffix = "1"

        URM_train_path = self.dataFile.extract(local_dataset_name + "/cf-train-{}-users.dat".format(train_interactions_file_suffix),
                                              path=self.decompressed_zip_file_folder + "decompressed/")

        URM_test_path = self.dataFile.extract(local_dataset_name + "/cf-test-{}-users.dat".format(train_interactions_file_suffix),
                                              path=self.decompressed_zip_file_folder + "decompressed/")

        ICM_path = self.dataFile.extract(local_dataset_name + "/mult_nor.mat",
                                              path=self.decompressed_zip_file_folder + "decompressed/")


        URM_all_dataframe = pd.concat([self._load_data_file(URM_test_path),
                                       self._load_data_file(URM_train_path)])

        if self.dataset_variant == "a":
            ICM_title_abstract = scipy.io.loadmat(ICM_path)['X']

        else:
            # Variant "t" uses a different file format and is transposed
            ICM_title_abstract = h5py.File(ICM_path,'r').get('X')
            ICM_title_abstract = sps.csr_matrix(ICM_title_abstract).T

        ICM_title_abstract = sps.coo_matrix(ICM_title_abstract)

        ICM_title_abstract_dataframe = pd.DataFrame({"ItemID": [str(x) for x in ICM_title_abstract.row],
                                                     "FeatureID": [str(x) for x in ICM_title_abstract.col],
                                                     "Data":  [x for x in ICM_title_abstract.data],
                                                     })

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_ICM(ICM_title_abstract_dataframe, "ICM_title_abstract")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)


        self._print("Cleaning Temporary Files")

        shutil.rmtree(self.decompressed_zip_file_folder + "decompressed/", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset





    def _load_data_file(self, filePath, separator = " "):

        fileHandle = open(filePath, "r")
        user_lists = []

        for line in fileHandle:

            line = line.replace("\n", "")
            line = line.split(separator)

            assert int(line[0]) == len(line[1:])

            user_lists.append(line[1:])

        fileHandle.close()

        # Split TagList in order to obtain a dataframe with a tag per row
        URM_all_dataframe = pd.DataFrame(user_lists, index = [str(x) for x in range(len(user_lists))]).stack()
        URM_all_dataframe = URM_all_dataframe.reset_index()[["level_0", 0]]
        URM_all_dataframe.columns = ['UserID', 'ItemID']
        URM_all_dataframe["Data"] = 1


        return  URM_all_dataframe





class CiteULike_aReader(_CiteULikeReader):

    DATASET_SUBFOLDER = "CiteULike_a/"

    def __init__(self, **kwargs):
        super(CiteULike_aReader, self).__init__(**kwargs)

        self.dataset_variant = "a"




class CiteULike_tReader(_CiteULikeReader):

    DATASET_SUBFOLDER = "CiteULike_t/"

    def __init__(self, **kwargs):
        super(CiteULike_tReader, self).__init__(**kwargs)

        self.dataset_variant = "t"