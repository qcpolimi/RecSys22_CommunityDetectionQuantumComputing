#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/01/2018

@author: Anonymous
"""

import scipy.sparse as sps
from recsys.Data_manager.DataPostprocessing import DataPostprocessing
import numpy as np



class DataPostprocessing_Anonymize(DataPostprocessing):
    """
    This class shuffles all the indices of URM, ICM and UCM in such a way to obtain an equivalent dataset with
    anonymized identifiers for users, items and features.
    """


    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: original or k_cores etc...
        """

        subfolder_name = "anonymize/"

        inner_subfolder_name = self.dataReader_object._get_dataset_name_data_subfolder()

        # Avoid concatenating the original/ part
        if inner_subfolder_name != self.DATASET_SUBFOLDER_ORIGINAL:
            subfolder_name += inner_subfolder_name

        return subfolder_name



    def _load_from_original_file(self):
        """
        _load_from_original_file will call the load of the dataset and then apply on it the k-cores
        :return:
        """

        loaded_dataset = self.dataReader_object.load_data()

        n_items = len(loaded_dataset.item_original_ID_to_index)
        n_users = len(loaded_dataset.user_original_ID_to_index)

        # Shuffle ITEMS and USERS in URM
        # The remapping is done as follows: shuffled_items[current_index] -> new index
        shuffled_items = np.arange(n_items, dtype = np.int)
        np.random.shuffle(shuffled_items)

        shuffled_users = np.arange(n_users, dtype = np.int)
        np.random.shuffle(shuffled_users)

        loaded_dataset.item_original_ID_to_index = {original_ID:shuffled_items[previous_index] for original_ID,previous_index in loaded_dataset.item_original_ID_to_index.items()}
        loaded_dataset.user_original_ID_to_index = {original_ID:shuffled_users[previous_index] for original_ID,previous_index in loaded_dataset.user_original_ID_to_index.items()}

        for URM_name, URM_object in loaded_dataset.AVAILABLE_URM.items():
            URM_object = sps.coo_matrix(URM_object)
            URM_object.row = shuffled_users[URM_object.row]
            URM_object.col = shuffled_items[URM_object.col]

            URM_object = sps.csr_matrix(URM_object)
            URM_object.sort_indices()
            loaded_dataset.AVAILABLE_URM[URM_name] = URM_object



        for ICM_name, ICM_object in loaded_dataset.AVAILABLE_ICM.items():

            ICM_feature_mapper = loaded_dataset.AVAILABLE_ICM_feature_mapper[ICM_name]

            n_features = len(ICM_feature_mapper)
            shuffled_features = np.arange(n_features, dtype = np.int)
            np.random.shuffle(shuffled_features)

            ICM_feature_mapper = {original_ID:shuffled_features[previous_index] for original_ID,previous_index in ICM_feature_mapper.items()}
            loaded_dataset.AVAILABLE_ICM_feature_mapper[ICM_name] = ICM_feature_mapper

            ICM_object = sps.coo_matrix(ICM_object)
            ICM_object.row = shuffled_items[ICM_object.row]
            ICM_object.col = shuffled_features[ICM_object.col]

            ICM_object = sps.csr_matrix(ICM_object)
            ICM_object.sort_indices()
            loaded_dataset.AVAILABLE_ICM[ICM_name] = ICM_object



        for UCM_name, UCM_object in loaded_dataset.AVAILABLE_UCM.items():

            UCM_feature_mapper = loaded_dataset.AVAILABLE_UCM_feature_mapper[UCM_name]

            n_features = len(UCM_feature_mapper)
            shuffled_features = np.arange(n_features, dtype = np.int)
            np.random.shuffle(shuffled_features)

            UCM_feature_mapper = {original_ID:shuffled_features[previous_index] for original_ID,previous_index in UCM_feature_mapper.items()}
            loaded_dataset.AVAILABLE_UCM_feature_mapper[UCM_name] = UCM_feature_mapper

            UCM_object = sps.coo_matrix(UCM_object)
            UCM_object.row = shuffled_users[UCM_object.row]
            UCM_object.col = shuffled_features[UCM_object.col]

            UCM_object = sps.csr_matrix(UCM_object)
            UCM_object.sort_indices()
            loaded_dataset.AVAILABLE_UCM[UCM_name] = UCM_object



        return loaded_dataset

