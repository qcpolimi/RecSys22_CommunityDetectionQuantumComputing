#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/12/19

@author: Anonymous
"""

import time, sys
import numpy as np
from enum import Enum
from recsys.Utils.seconds_to_biggest_unit import seconds_to_biggest_unit


from recsys.Evaluation.Evaluator import EvaluatorHoldout, EvaluatorNegativeItemSample

from recsys.Evaluation.metrics_compare_models import CommonHits, CommonItems, CommonHitsQuota



class EvaluatorCompareModelsMetrics(Enum):
    COMMON_IN_LIST = "COMMON_IN_LIST"
    COMMON_HITS = "COMMON_HITS"
    COMMON_HITS_QUOTA = "COMMON_HITS_QUOTA"




def _create_empty_metrics_dict(cutoff_list, n_items, n_users, URM_train, URM_test, ignore_items, ignore_users, diversity_similarity_object):

    empty_dict = {}

    for cutoff in cutoff_list:

        cutoff_dict = {}

        for metric in EvaluatorCompareModelsMetrics:

            if metric == EvaluatorCompareModelsMetrics.COMMON_IN_LIST:
                cutoff_dict[metric.value] = CommonItems(cutoff)

            elif metric == EvaluatorCompareModelsMetrics.COMMON_HITS:
                cutoff_dict[metric.value] = CommonHits()

            elif metric == EvaluatorCompareModelsMetrics.COMMON_HITS_QUOTA:
                cutoff_dict[metric.value] = CommonHitsQuota()
            else:
                cutoff_dict[metric.value] = 0.0

        empty_dict[cutoff] = cutoff_dict

    return  empty_dict




class _BaseEvaluatorCompareModels(object):


    EVALUATOR_NAME = "BaseEvaluatorCompareModels"


    def _compute_compare_metrics_on_recommendation_list(self, test_user_batch_array,
                                                base_recommended_items_batch_list, base_scores_batch,
                                                other_recommended_items_batch_list, other_scores_batch,
                                                results_dict):

        
        assert len(other_recommended_items_batch_list) == len(base_recommended_items_batch_list), "{}: recommended_items_batch_list shapes for the base model and other model are different {}, {}".format(
            self.EVALUATOR_NAME, len(other_recommended_items_batch_list), len(base_recommended_items_batch_list))
        
        assert other_scores_batch.shape == base_scores_batch.shape, "{}: scores_batch shapes for the base model and other model are different {}, {}".format(
            self.EVALUATOR_NAME, other_scores_batch.shape, base_scores_batch.shape)
        
        
        
        assert len(other_recommended_items_batch_list) == len(test_user_batch_array), "{}: recommended_items_batch_list contained recommendations for {} users, expected was {}".format(
            self.EVALUATOR_NAME, len(other_recommended_items_batch_list), len(test_user_batch_array))

        assert base_scores_batch.shape[0] == len(test_user_batch_array), "{}: scores_batch contained scores for {} users, expected was {}".format(
            self.EVALUATOR_NAME, base_scores_batch.shape[0], len(test_user_batch_array))

        assert base_scores_batch.shape[1] == self.n_items, "{}: scores_batch contained scores for {} items, expected was {}".format(
            self.EVALUATOR_NAME, base_scores_batch.shape[1], self.n_items)


        # Compute recommendation quality for each user in batch
        for batch_user_index in range(len(other_recommended_items_batch_list)):

            test_user = test_user_batch_array[batch_user_index]
            relevant_items = self.get_user_relevant_items(test_user)

            # Being the URM CSR, the indices are the non-zero column indexes
            other_recommended_items = other_recommended_items_batch_list[batch_user_index]
            other_recommended_items = np.array(other_recommended_items)
            other_is_relevant = np.in1d(other_recommended_items, relevant_items, assume_unique=True)
            
            # Being the URM CSR, the indices are the non-zero column indexes
            base_recommended_items = base_recommended_items_batch_list[batch_user_index]
            base_recommended_items = np.array(base_recommended_items)
            base_is_relevant = np.in1d(base_recommended_items, relevant_items, assume_unique=True)

            self._n_users_evaluated += 1

            for cutoff in self.cutoff_list:

                results_current_cutoff = results_dict[cutoff]

                other_is_relevant_current_cutoff = other_is_relevant[0:cutoff]
                other_recommended_items_current_cutoff = other_recommended_items[0:cutoff]

                base_is_relevant_current_cutoff = base_is_relevant[0:cutoff]
                base_recommended_items_current_cutoff = base_recommended_items[0:cutoff]

                
                common_items_flag = np.in1d(other_recommended_items_current_cutoff,
                                            base_recommended_items_current_cutoff,
                                            assume_unique=True)
                
                common_recommended_items_cutoff = other_recommended_items_current_cutoff[common_items_flag]
                common_is_relevant_cutoff = np.in1d(common_recommended_items_cutoff, relevant_items, assume_unique=True)

                results_current_cutoff[EvaluatorCompareModelsMetrics.COMMON_IN_LIST.value].add_recommendations(common_recommended_items_cutoff)
                results_current_cutoff[EvaluatorCompareModelsMetrics.COMMON_HITS_QUOTA.value].add_recommendations(other_is_relevant_current_cutoff, base_is_relevant_current_cutoff, common_is_relevant_cutoff)
                results_current_cutoff[EvaluatorCompareModelsMetrics.COMMON_HITS.value].add_recommendations(common_is_relevant_cutoff)


            if time.time() - self._start_time_print > 30 or self._n_users_evaluated==len(self.users_to_evaluate):

                elapsed_time = time.time()-self._start_time
                new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

                self._print("Processed {} ({:4.1f}%) in {:.2f} {}. Users per second: {:.0f}".format(
                              self._n_users_evaluated,
                              100.0* float(self._n_users_evaluated)/len(self.users_to_evaluate),
                              new_time_value, new_time_unit,
                              float(self._n_users_evaluated)/elapsed_time))

                sys.stdout.flush()
                sys.stderr.flush()

                self._start_time_print = time.time()



        return results_dict


    def compareRecommenders(self, base_recommender_object, other_recommender_object):
        
        self.base_recommender_object = base_recommender_object

        return self.evaluateRecommender(other_recommender_object)
    
    
 


class EvaluatorCompareModelsHoldout(EvaluatorHoldout, _BaseEvaluatorCompareModels):
    """EvaluatorHoldout"""

    EVALUATOR_NAME = "EvaluatorCompareModelsHoldout"
   
    def _run_evaluation_on_selected_users(self, other_recommender_object, users_to_evaluate, block_size = None):

        if block_size is None:
            block_size = min(1000, int(1e8/self.n_items))
            block_size = min(block_size, len(users_to_evaluate))


        results_dict = _create_empty_metrics_dict(self.cutoff_list,
                                                  self.n_items, self.n_users,
                                                  other_recommender_object.get_URM_train(),
                                                  self.URM_test,
                                                  self.ignore_items_ID,
                                                  self.ignore_users_ID,
                                                  self.diversity_object)


        if self.ignore_items_flag:
            self.base_recommender_object.set_items_to_ignore(self.ignore_items_ID)
            other_recommender_object.set_items_to_ignore(self.ignore_items_ID)

        # Start from -block_size to ensure it to be 0 at the first block
        user_batch_start = 0
        user_batch_end = 0

        while user_batch_start < len(users_to_evaluate):

            user_batch_end = user_batch_start + block_size
            user_batch_end = min(user_batch_end, len(users_to_evaluate))

            test_user_batch_array = np.array(users_to_evaluate[user_batch_start:user_batch_end])
            user_batch_start = user_batch_end

            # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
            other_recommended_items_batch_list, other_scores_batch = other_recommender_object.recommend(test_user_batch_array,
                                                                                            remove_seen_flag=self.exclude_seen,
                                                                                            cutoff = self.max_cutoff,
                                                                                            remove_top_pop_flag=False,
                                                                                            remove_custom_items_flag=self.ignore_items_flag,
                                                                                            return_scores = True
                                                                                            )

            # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
            base_recommended_items_batch_list, base_scores_batch = self.base_recommender_object.recommend(test_user_batch_array,
                                                                                            remove_seen_flag=self.exclude_seen,
                                                                                            cutoff = self.max_cutoff,
                                                                                            remove_top_pop_flag=False,
                                                                                            remove_custom_items_flag=self.ignore_items_flag,
                                                                                            return_scores = True
                                                                                            )
            
            results_dict = self._compute_compare_metrics_on_recommendation_list(test_user_batch_array = test_user_batch_array,
                                                         base_recommended_items_batch_list = base_recommended_items_batch_list,
                                                         base_scores_batch = base_scores_batch,
                                                         other_recommended_items_batch_list = other_recommended_items_batch_list,
                                                         other_scores_batch = other_scores_batch,
                                                         results_dict = results_dict)


        return results_dict





class EvaluatorCompareModelsNegativeItemSample(EvaluatorNegativeItemSample, _BaseEvaluatorCompareModels):
    """EvaluatorNegativeItemSample"""

    EVALUATOR_NAME = "EvaluatorCompareModelsNegativeItemSample"


    def _get_user_specific_items_to_compute(self, user_id):

        start_pos = self.URM_items_to_rank.indptr[user_id]
        end_pos = self.URM_items_to_rank.indptr[user_id+1]

        items_to_compute = self.URM_items_to_rank.indices[start_pos:end_pos]

        return items_to_compute


    def _run_evaluation_on_selected_users(self, other_recommender_object, users_to_evaluate, block_size = None):

        if block_size is None:
            block_size = min(1000, int(1e8/self.n_items))
            block_size = min(block_size, len(users_to_evaluate))


        results_dict = _create_empty_metrics_dict(self.cutoff_list,
                                                  self.n_items, self.n_users,
                                                  other_recommender_object.get_URM_train(),
                                                  self.URM_test,
                                                  self.ignore_items_ID,
                                                  self.ignore_users_ID,
                                                  self.diversity_object)


        if self.ignore_items_flag:
            self.base_recommender_object.set_items_to_ignore(self.ignore_items_ID)
            other_recommender_object.set_items_to_ignore(self.ignore_items_ID)


        for test_user in users_to_evaluate:

            items_to_compute = self._get_user_specific_items_to_compute(test_user)

            test_user_batch_array = np.atleast_1d(test_user)
            
            # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
            other_recommended_items_batch_list, other_scores_batch = other_recommender_object.recommend(test_user_batch_array,
                                                                                            remove_seen_flag=self.exclude_seen,
                                                                                            cutoff = self.max_cutoff,
                                                                                            remove_top_pop_flag=False,
                                                                                            items_to_compute = items_to_compute,
                                                                                            remove_custom_items_flag=self.ignore_items_flag,
                                                                                            return_scores = True
                                                                                            )

            # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
            base_recommended_items_batch_list, base_scores_batch = self.base_recommender_object.recommend(test_user_batch_array,
                                                                                            remove_seen_flag=self.exclude_seen,
                                                                                            cutoff = self.max_cutoff,
                                                                                            remove_top_pop_flag=False,
                                                                                            items_to_compute = items_to_compute,
                                                                                            remove_custom_items_flag=self.ignore_items_flag,
                                                                                            return_scores = True
                                                                                            )
            
            results_dict = self._compute_compare_metrics_on_recommendation_list(test_user_batch_array = test_user_batch_array,
                                                         base_recommended_items_batch_list = base_recommended_items_batch_list,
                                                         base_scores_batch = base_scores_batch,
                                                         other_recommended_items_batch_list = other_recommended_items_batch_list,
                                                         other_scores_batch = other_scores_batch,
                                                         results_dict = results_dict)


        return results_dict

