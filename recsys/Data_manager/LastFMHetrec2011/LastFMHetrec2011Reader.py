#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Anonymous
"""


import zipfile, shutil
from recsys.Data_manager.DataReader import DataReader
from recsys.Data_manager.DataReader_utils import download_from_URL
import pandas as pd
from recsys.Data_manager.DatasetMapperManager import DatasetMapperManager


class LastFMHetrec2011Reader(DataReader):

    DATASET_URL = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
    DATASET_SUBFOLDER = "LastFMHetrec2011/"
    AVAILABLE_URM = ["URM_all", "URM_occurrence"]
    AVAILABLE_ICM = ["ICM_tags", "ICM_tags_count"]



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        folder_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            dataFile = zipfile.ZipFile(folder_path + "hetrec2011-lastfm-2k.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find or extract data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, folder_path, "hetrec2011-lastfm-2k.zip")

            dataFile = zipfile.ZipFile(folder_path + "hetrec2011-lastfm-2k.zip")



        URM_path = dataFile.extract("user_artists.dat", path=folder_path + "decompressed")
        tags_path = dataFile.extract("user_taggedartists-timestamps.dat", path=folder_path + "decompressed")


        self._print("Loading Interactions")
        URM_occurrence_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep="\t", header=0,
                                        dtype={0:str, 1:str, 2:int})

        URM_occurrence_dataframe.columns = ["UserID", "ItemID", "Data"]

        URM_all_dataframe = URM_occurrence_dataframe.copy()
        URM_all_dataframe["Data"] = 1

        self._print("Loading Item Features")
        ICM_tags_count_dataframe = pd.read_csv(filepath_or_buffer=tags_path, sep="\t", header=0,
                                         dtype={0:str, 1:str, 2:str, 3:float})

        ICM_tags_count_dataframe.columns = ["UserID", "ItemID", "FeatureID", "Timestamp"]

        ICM_tags_count_dataframe = ICM_tags_count_dataframe[["ItemID", "FeatureID"]]
        ICM_tags_count_dataframe["Data"] = 1

        # The dataset contains duplicated tags, one ICM contains the counter the other only 1
        ICM_tags_count_dataframe = ICM_tags_count_dataframe.groupby(["ItemID","FeatureID"],as_index=False)["Data"].sum()

        ICM_tags_dataframe = ICM_tags_count_dataframe.copy()
        ICM_tags_dataframe["Data"] = 1


        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_occurrence_dataframe, "URM_occurrence")
        dataset_manager.add_ICM(ICM_tags_dataframe, "ICM_tags")
        dataset_manager.add_ICM(ICM_tags_count_dataframe, "ICM_tags_count")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)


        self._print("Cleaning Temporary Files")

        shutil.rmtree(folder_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset
