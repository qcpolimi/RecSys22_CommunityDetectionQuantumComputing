#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/12/18

@author: Anonymous
"""

import time
from recsys.Evaluation.Evaluator import get_result_string


class Recommender_k_Fold_Wrapper():

     RECOMMENDER_NAME = "Recommender_k_Fold_Wrapper"

     def __init__(self, recommender_class, n_folds, recommender_input_args_list, verbose = True):

         assert n_folds is not None and n_folds>=1, \
            "{}: number of folds must be an integer value greater than 1, provided value was '{}'".format(self.RECOMMENDER_NAME, str(n_folds))

         assert n_folds == len(recommender_input_args_list), \
            "{}: inconsistent values for number of folds '{}' and the length of constructor_data_list '{}'".format(self.RECOMMENDER_NAME, n_folds, len(recommender_input_args_list))

         self.n_folds = n_folds
         self.verbose = verbose
         self.recommender_class = recommender_class
         self.RECOMMENDER_NAME = self.recommender_class.RECOMMENDER_NAME + "_k_Fold_Wrapper"
         self.recommender_input_args_list = recommender_input_args_list

         self._recommender_instance_list = [None] * self.n_folds
         self._recommender_fit_time_list = [0.0] * self.n_folds


         for current_fold in range(self.n_folds):

             self._print("Building fold {} of {}".format(current_fold+1, self.n_folds))

             recommender_input_args = self.recommender_input_args_list[current_fold]

             start_time = time.time()

             recommender_instance = self.recommender_class(
                 *recommender_input_args.CONSTRUCTOR_POSITIONAL_ARGS,
                 **recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS)

             train_time = time.time() - start_time

             self._recommender_instance_list[current_fold] = recommender_instance
             self._recommender_fit_time_list[current_fold] += train_time



     def _print(self, string):
        if self.verbose:
            print("{}: {}".format(self.RECOMMENDER_NAME, string))


     def get_n_folds(self):
         return self.n_folds


     def fit(self, *posargs, **kwargs):

         for current_fold in range(self.n_folds):
             self._print("Fitting fold {} of {}".format(current_fold+1, self.n_folds))

             recommender_instance = self._recommender_instance_list[current_fold]

             start_time = time.time()
             recommender_instance.fit(*posargs, **kwargs)

             train_time = time.time() - start_time
             self._recommender_fit_time_list[current_fold] += train_time


     def set_recommender_fold(self, recommender_instance, fold_index):
        self._recommender_instance_list[fold_index] = recommender_instance


     def get_recommender_fold(self, fold_index):
         return self._recommender_instance_list[fold_index]


     def get_recommender_fit_time_fold(self, fold_index):
         return self._recommender_fit_time_list[fold_index]


     def save_model(self, folder_path, file_name = None):

         assert file_name is not None, "{}: file_name must not be None".format(self.RECOMMENDER_NAME)

         for fold_index in range(self.n_folds):

            recommender_instance = self._recommender_instance_list[fold_index]
            recommender_instance.save_model(folder_path, file_name + "_fold_{}".format(fold_index))




     def load_model(self, recommender_instance_list, folder_path, file_name = None):

         assert file_name is not None, \
             "{}: file_name must not be None".format(self.RECOMMENDER_NAME)

         assert len(recommender_instance_list) == self.n_folds,\
             "{}: recommender_instance_list must be as long as the number of folds".format(self.RECOMMENDER_NAME)

         for fold_index in range(self.n_folds):

            recommender_instance = recommender_instance_list[fold_index]

            recommender_instance.load_model(folder_path, file_name + "_fold_{}".format(fold_index))
            self._recommender_instance_list[fold_index] = recommender_instance



class Evaluator_k_Fold_Wrapper():
    """Evaluator_k_Fold_Wrapper"""

    EVALUATOR_NAME = "Evaluator_k_Fold_Wrapper"

    def __init__(self, evaluator_instance_list, n_folds, verbose = True):

         assert n_folds is not None and n_folds>=1, \
            "{}: number of folds must be an integer value greater than 1, provided value was '{}'".format(self.EVALUATOR_NAME, str(n_folds))

         assert n_folds == len(evaluator_instance_list), \
            "{}: inconsistent values for number of folds '{}' and the length of evaluator_instance_list '{}'".format(self.EVALUATOR_NAME, n_folds, len(evaluator_instance_list))

         self.evaluator_instance_list = evaluator_instance_list
         self.n_folds = len(evaluator_instance_list)
         self.verbose = verbose

         self._recommender_evaluation_time_list = [None] * self.n_folds
         self._recommender_evaluation_result_list = [None] * self.n_folds



    def _print(self, string):
        if self.verbose:
            print("{}: {}".format(self.EVALUATOR_NAME, string))

    def get_n_folds(self):
         return self.n_folds


    def set_evaluator_fold(self, evaluator_instance, fold_index):
        self.evaluator_instance_list[fold_index] = evaluator_instance


    def get_evaluator_fold(self, fold_index):
         return self.evaluator_instance_list[fold_index]

    def get_recommender_evaluation_time_fold(self, fold_index):
         return self._recommender_evaluation_time_list[fold_index]

    def get_recommender_evaluation_result_fold(self, fold_index):
         return self._recommender_evaluation_result_list[fold_index]

    def evaluateRecommender(self, recommender_object_k_fold):
        """
        :param recommender_object_k_fold: the trained recommender object, a BaseRecommender subclass
        """

        assert isinstance(recommender_object_k_fold, Recommender_k_Fold_Wrapper),\
            "{}: Recommender object is not instance of Recommender_k_Fold_Wrapper".format(self.EVALUATOR_NAME)

        assert recommender_object_k_fold.get_n_folds() == self.n_folds,\
            "{}: Recommender object has '{}' folds while Evaluator has '{}'".format(self.EVALUATOR_NAME, recommender_object_k_fold.get_n_folds(), self.n_folds)


        global_result_dict = None

        for current_fold in range(self.n_folds):

            self._print("Evaluating fold {} of {}".format(current_fold+1, self.n_folds))

            evaluator_fold = self.get_evaluator_fold(current_fold)
            recommender_fold = recommender_object_k_fold.get_recommender_fold(current_fold)

            start_time = time.time()

            fold_result_dict, result_string = evaluator_fold.evaluateRecommender(recommender_fold)

            evaluation_time = time.time() - start_time

            self._recommender_evaluation_time_list[current_fold] = evaluation_time
            self._recommender_evaluation_result_list[current_fold] = fold_result_dict

            if global_result_dict is None:
                global_result_dict = {}
                for cutoff in fold_result_dict.keys():
                    global_result_dict[cutoff] = {}
                    for metric in fold_result_dict[cutoff].keys():
                        global_result_dict[cutoff][metric] = fold_result_dict[cutoff][metric] / self.n_folds

            else:
                for cutoff in fold_result_dict.keys():
                    for metric in fold_result_dict[cutoff].keys():
                        global_result_dict[cutoff][metric] += fold_result_dict[cutoff][metric] / self.n_folds

        results_run_string = get_result_string(global_result_dict)

        return (global_result_dict, results_run_string)




from recsys.HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from recsys.Evaluation.Evaluator import EvaluatorHoldout



def get_k_fold_bayesian_search_struct_data(dataSplitter_k_fold, recommender_class, n_folds):


    recommender_constructor_data_list = []
    recommender_constructor_data_last_list = []
    evaluator_list = []

    for fold_index, dataSplitter_fold in enumerate(dataSplitter_k_fold):

        print("Processing fold {}".format(fold_index))

        URM_train, URM_validation, URM_test = dataSplitter_fold.get_holdout_split()

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {}
        )

        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation

        recommender_constructor_data_list.append(recommender_input_args)
        recommender_constructor_data_last_list.append(recommender_input_args_last_test)

        evaluator = EvaluatorHoldout(URM_test, [10], exclude_seen=False)
        evaluator_list.append(evaluator)


    evaluator_k_Fold = Evaluator_k_Fold_Wrapper(evaluator_list, n_folds)


    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [recommender_class, n_folds, recommender_constructor_data_list],
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {}
    )

    recommender_input_args_last = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [recommender_class, n_folds, recommender_constructor_data_last_list],
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {}
    )


    return recommender_input_args, recommender_input_args_last, evaluator_k_Fold
















#
#
# class SearchBayesianSkopt_k_fold(SearchBayesianSkopt):
#
#     ALGORITHM_NAME = "SearchBayesianSkopt_k_fold"
#
#
#     def __init__(self, recommender_class, evaluator_validation_list = None, evaluator_test_list = None, n_folds = None):
#
#         assert n_folds is not None and n_folds>=1, \
#             "{}: number of folds must be an integer value greater than 1, provided value was '{}'".format(self.ALGORITHM_NAME, str(n_folds))
#
#         self.n_folds = n_folds
#
#
#         self.recommender_class = recommender_class
#
#         self.evaluator_validation = [None]*self.n_folds
#         self.evaluator_test = [None]*self.n_folds
#
#         self.results_test_best = {}
#         self.parameter_dictionary_best = {}
#
#         if evaluator_validation_list is None:
#             raise ValueError("{}: evaluator_validation must be provided".format(self.ALGORITHM_NAME))
#         else:
#             for fold_index in range(self.n_folds):
#                 self.evaluator_validation[fold_index] = evaluator_validation_list[fold_index]
#
#         if evaluator_test_list is None:
#             self.evaluator_test = None
#         else:
#             for fold_index in range(self.n_folds):
#                 self.evaluator_test[fold_index] = evaluator_test_list[fold_index]
#
#
#
#
#     def _init_metadata_dict(self):
#
#         super(SearchBayesianSkopt_k_fold, self)._init_metadata_dict()
#
#         for fold_index in range(self.n_folds):
#             self.metadata_dict["validation_result_list_fold_{}".format(fold_index)] = [None]*self.n_calls
#             self.metadata_dict["test_result_list_fold_{}".format(fold_index)] = [None]*self.n_calls
#
#
#
#
#     def _evaluate_on_validation(self, current_fit_hyperparameters):
#
#         global_result_dict = None
#         global_train_time = 0.0
#         global_evaluation_time = 0.0
#
#         recommender5fold_wrapper = Recommender5Fold_Wrapper(n_folds=self.n_folds)
#
#         for fold_index in range(self.n_folds):
#
#             start_time = time.time()
#
#             # Construct a new recommender instance
#             recommender_instance = self.recommender_class(*self.recommender_input_args.CONSTRUCTOR_POSITIONAL_ARGS[fold_index],
#                                                           **self.recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS[fold_index])
#
#
#             print("{}: Testing Fold {} Config:".format(self.ALGORITHM_NAME, fold_index, current_fit_hyperparameters))
#
#
#             recommender_instance.fit(*self.recommender_input_args.FIT_POSITIONAL_ARGS,
#                                      **self.recommender_input_args.FIT_KEYWORD_ARGS,
#                                      **current_fit_hyperparameters,
#                                      **self.hyperparams_single_value)
#
#             fold_train_time = time.time() - start_time
#             start_time = time.time()
#
#             # Evaluate recommender and get results for the first cutoff
#             fold_evaluator_validation = self.evaluator_validation[fold_index]
#
#             fold_result_dict, result_string = fold_evaluator_validation.evaluateRecommender(recommender_instance, self.recommender_input_args)
#
#             first_cutoff = list(fold_result_dict.keys())[0]
#             fold_result_dict = fold_result_dict[first_cutoff]
#
#             fold_evaluation_time = time.time() - start_time
#
#
#             recommender5fold_wrapper.set_recommender_instance(recommender_instance, fold_index)
#
#
#             if self.save_metadata:
#                 self.metadata_dict["validation_result_list_fold_{}".format(fold_index)][self.model_counter] = fold_result_dict.copy()
#
#
#
#             if global_result_dict is None:
#                 global_result_dict = {}
#                 for metric in fold_result_dict.keys():
#                     global_result_dict[metric] = fold_result_dict[metric]/self.n_folds
#
#             else:
#                 for metric in fold_result_dict.keys():
#                     global_result_dict[metric] += fold_result_dict[metric]/self.n_folds
#
#
#             global_train_time += fold_train_time/self.n_folds
#             global_evaluation_time += fold_evaluation_time/self.n_folds
#
#
#         global_result_string = get_result_string({first_cutoff: global_result_dict})
#
#
#         return global_result_dict, global_result_string, recommender5fold_wrapper, global_train_time, global_evaluation_time
#
#
#
#
#
#
#
#     def _evaluate_on_test(self, recommender_instance):
#
#
#         global_result_dict = None
#         global_evaluation_test_time = 0.0
#
#
#         for fold_index in range(self.n_folds):
#
#             fold_recommender_instance = recommender_instance.get_recommender_instance(fold_index)
#
#             start_time = time.time()
#
#             # Evaluate recommender and get results for the first cutoff
#             fold_evaluator_test = self.evaluator_test[fold_index]
#
#             fold_result_dict, _ = fold_evaluator_test.evaluateRecommender(fold_recommender_instance, self.recommender_input_args)
#
#             if self.save_metadata:
#                 self.metadata_dict["test_result_list_fold_{}".format(fold_index)][self.model_counter] = fold_result_dict.copy()
#
#
#             fold_evaluation_test_time = time.time() - start_time
#
#             if global_result_dict is None:
#                 global_result_dict = {}
#                 for cutoff in fold_result_dict.keys():
#                     global_result_dict[cutoff] = {}
#                     for metric in fold_result_dict[cutoff].keys():
#                         global_result_dict[cutoff][metric] = fold_result_dict[cutoff][metric]/self.n_folds
#
#             else:
#                 for cutoff in fold_result_dict.keys():
#                     for metric in fold_result_dict[cutoff].keys():
#                         global_result_dict[cutoff][metric] += fold_result_dict[cutoff][metric]/self.n_folds
#
#
#
#             global_evaluation_test_time += fold_evaluation_test_time/self.n_folds
#
#
#
#         global_result_string = get_result_string(global_result_dict)
#
#         writeLog("{}: Best result evaluated on URM_test. Config: {} - results:\n{}\n".format(self.ALGORITHM_NAME, self.best_solution_parameters, global_result_string), self.log_file)
#
#         return global_result_dict, global_evaluation_test_time
#
#
