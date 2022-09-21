#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/02/20

@author: Anonymous
"""




import numpy as np
import unittest, shutil


from recsys.Recommenders.NonPersonalizedRecommender import TopPop
from recsys.Data_manager import *


class MyTestCase(unittest.TestCase):

    def test_compute_item_score(self):

        recommender_instance, URM_train, URM_test = get_data_and_rec_instance(TopPop, Movielens100KReader)

        n_users, n_items = URM_train.shape

        recommender_instance.fit()

        user_batch_size = 1000

        user_id_list = np.arange(n_users, dtype=np.int)
        np.random.shuffle(user_id_list)
        user_id_list = user_id_list[:user_batch_size]


        item_scores = recommender_instance._compute_item_score(user_id_array = user_id_list,
                                                               items_to_compute = None)

        user_batch_size = min(user_batch_size, n_users)

        self.assertEqual(item_scores.shape, (user_batch_size, n_items), "item_scores shape not correct, contains more users than in user_id_array")
        self.assertFalse(np.any(np.isnan(item_scores)), "item_scores contains np.nan values")


        item_batch_size = 500

        item_id_list = np.arange(n_items, dtype=np.int)
        np.random.shuffle(item_id_list)
        item_id_list = item_id_list[:item_batch_size]

        item_scores = recommender_instance._compute_item_score(user_id_array = user_id_list,
                                                               items_to_compute = item_id_list)

        self.assertEqual(item_scores.shape, (user_batch_size, n_items), "item_scores shape not correct, does not contain all items")
        self.assertFalse(np.any(np.isnan(item_scores)), "item_scores contains np.nan values")
        self.assertFalse(np.any(np.isposinf(item_scores)), "item_scores contains +np.inf values")

        #Check items not in list have a score of -np.inf
        item_id_not_to_compute = np.ones(n_items, dtype=np.bool)
        item_id_not_to_compute[item_id_list] = False

        self.assertTrue(np.all(np.isneginf(item_scores[:,item_id_not_to_compute])), "item_scores contains scores for items that should not be computed")




    def test_save_and_load(self):

        recommender_class = TopPop

        recommender_instance_original, URM_train, URM_test = get_data_and_rec_instance(recommender_class, Movielens100KReader)
        n_users, n_items = URM_train.shape


        from recsys.Evaluation.Evaluator import EvaluatorHoldout

        evaluator_test = EvaluatorHoldout(URM_test, [50], exclude_seen=True)


        folder_path = "./temp_folder/"
        file_name="temp_file"

        recommender_instance_original.fit()
        recommender_instance_original.save_model(folder_path=folder_path, file_name=file_name)

        results_run_original, _ = evaluator_test.evaluateRecommender(recommender_instance_original)



        recommender_instance_loaded = recommender_class(URM_train)
        recommender_instance_loaded.load_model(folder_path=folder_path, file_name=file_name)

        results_run_loaded, _ = evaluator_test.evaluateRecommender(recommender_instance_loaded)


        print("Result original: {}\n".format(results_run_original))
        print("Result loaded: {}\n".format(results_run_loaded))


        for user_id in range(n_users):

            item_scores_original = recommender_instance_original._compute_item_score(user_id_array = [user_id],
                                                               items_to_compute = None)

            item_scores_loaded = recommender_instance_loaded._compute_item_score(user_id_array = [user_id],
                                                           items_to_compute = None)

            self.assertTrue(np.allclose(item_scores_original, item_scores_loaded), "item_scores of the fitted model and of the loaded model are different")


        shutil.rmtree(folder_path, ignore_errors=True)





def get_data_and_rec_instance(recommender_class, dataset_class):

    from recsys.Data_manager.DataSplitter_Holdout import DataSplitter_Holdout

    dataset_object = dataset_class()
    dataSplitter = DataSplitter_Holdout(dataset_object, split_interaction_quota_list=[80, 10, 10])
    dataSplitter.load_data()

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

    URM_train = URM_train+URM_validation

    recommender_instance = recommender_class(URM_train)

    return recommender_instance, URM_train, URM_test






if __name__ == '__main__':



    unittest.main()

