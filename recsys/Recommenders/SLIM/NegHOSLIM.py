#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/11/2021

@author: Anonymous

Porting of:
@inbook{10.1145/3460231.3474273,
author = {Steck, Harald and Liang, Dawen},
title = {Negative Interactions for Improved Collaborative Filtering: Don’t Go Deeper, Go Higher},
year = {2021},
isbn = {9781450384582},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3460231.3474273},
abstract = { The recommendation-accuracy of collaborative filtering approaches is typically improved
when taking into account higher-order interactions [5, 6, 9, 10, 11, 16, 18, 24, 25,
28, 31, 34, 36, 41, 42, 44]. While deep nonlinear models are theoretically able to
learn higher-order interactions, their capabilities were, however, found to be quite
limited in practice [5]. Moreover, the use of low-dimensional embeddings in deep networks
may severely limit their expressiveness [8]. This motivated us in this paper to explore
a simple extension of linear full-rank models that allow for higher-order interactions
as additional explicit input-features. Interestingly, we observed that this model-class
obtained by far the best ranking accuracies on the largest data set in our experiments,
while it was still competitive with various state-of-the-art deep-learning models
on the smaller data sets. Moreover, our approach can also be interpreted as a simple
yet effective improvement of the (linear) HOSLIM [11] model: by simply removing the
constraint that the learned higher-order interactions have to be non-negative, we
observed that the accuracy-gains due to higher-order interactions more than doubled
in our experiments. The reason for this large improvement was that large positive
higher-order interactions (as used in HOSLIM [11]) are relatively infrequent compared
to the number of large negative higher-order interactions in the three well-known
data-sets used in our experiments. We further characterize the circumstances where
the higher-order interactions provide the most significant improvements.},
booktitle = {Fifteenth ACM Conference on Recommender Systems},
pages = {34–43},
numpages = {10}
}

"""

import time
import numpy as np
import scipy.sparse as sps
from recsys.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from recsys.Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from recsys.Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from utils.DataIO import DataIO
from copy import deepcopy


#
# ### functions to create the feature-pairs
# def create_list_feature_pairs(XtX, threshold):
#     AA = sps.triu(abs(XtX))
#     AA.setdiag(0.0)
#     AA = AA>threshold
#     AA.eliminate_zeros()
#     ii_pairs = AA.nonzero()
#     return ii_pairs
#
# def create_matrix_Z(ii_pairs, X):
#     MM = sps.lil_matrix((len(ii_pairs[0]), X.shape[1]),    dtype=np.float)
#     MM[np.arange(MM.shape[0]) , ii_pairs[0]   ]=1.0
#     MM[np.arange(MM.shape[0]) , ii_pairs[1]   ]=1.0
#
#     CCmask = 1.0-MM.todense()    # see Eq. 8 in the paper
#
#     MM=sps.csc_matrix(MM.T)
#     Z=  X * MM
#     Z= (Z == 2.0 )
#     Z=Z*1.0
#     return [ Z, CCmask]

### functions to create the feature-pairs
def create_list_feature_pairs(XtX, threshold):
    AA= np.triu(np.abs(XtX))
    AA[ np.diag_indices(AA.shape[0]) ]=0.0
    ii_pairs = np.where((AA>threshold)==True)
    return ii_pairs

def create_matrix_Z(ii_pairs, X):
    MM = np.zeros( (len(ii_pairs[0]), X.shape[1]),    dtype=np.float)
    MM[np.arange(MM.shape[0]) , ii_pairs[0]   ]=1.0
    MM[np.arange(MM.shape[0]) , ii_pairs[1]   ]=1.0
    CCmask = 1.0-MM    # see Eq. 8 in the paper
    MM=sps.csc_matrix(MM.T)
    Z=  X * MM
    Z= (Z == 2.0 )
    Z=Z*1.0
    return [ Z, CCmask]

#
# class NegHOSLIMRecommender(BaseItemSimilarityMatrixRecommender, Incremental_Training_Early_Stopping):
#     """
#     """
#
#     RECOMMENDER_NAME = "NegHOSLIMRecommender"
#
#     def __init__(self, URM_train, verbose = True):
#         super(NegHOSLIMRecommender, self).__init__(URM_train, verbose = verbose)
#
#
#
#     def _compute_item_score(self, user_id_array, items_to_compute = None):
#
#         Xtest = self.URM_train[user_id_array,:]
#         Ztest = self.Z[user_id_array,:]
#
#         if items_to_compute is not None:
#             item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
#             BB_items_to_compute = self.BB[items_to_compute,:][:,items_to_compute]
#             item_scores[:, items_to_compute] = (Xtest[:, items_to_compute]).dot(BB_items_to_compute) + \
#                                                Ztest.dot(self.CC[:, items_to_compute])
#
#         else:
#             item_scores = (Xtest).dot(self.BB) + Ztest.dot(self.CC)
#
#         return item_scores
#
#
#     def _create_feature_pairs(self):
#
#         self.X = self.URM_train
#
#         start_time = time.time()
#         self._print("Creating Feature Pairs...")
#
#         self.XtX = self.X.transpose() * self.X
#         self.XtXdiag=deepcopy(self.XtX.diagonal())
#
#         ### create the list of feature-pairs and the higher-order matrix Z
#         # self.XtX[ np.diag_indices(self.XtX.shape[0]) ] = self.XtXdiag #if code is re-run, ensure that the diagonal is correct
#         ii_feature_pairs = create_list_feature_pairs(self.XtX, self.threshold)
#
#         # print("number of feature-pairs: {}".format(len(ii_feature_pairs[0])))
#         self.Z, self.CCmask = create_matrix_Z(ii_feature_pairs, self.X)
#         # Z_test_data_tr , _ = create_matrix_Z(ii_feature_pairs, test_data_tr)
#
#         new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
#         self._print("Creating Feature Pairs... done in {:.2f} {}. Number of Feature-Pairs: {}".format( new_time_value, new_time_unit, len(ii_feature_pairs[0])))
#
#
#
#     def fit(self, epochs=300, threshold = 3.5e4, lambdaBB = 5e2, lambdaCC = 5e3, rho = 1e5,
#             **earlystopping_kwargs):
#
#         self.rho = rho
#         self.threshold = threshold
#
#         ####### Data structures summary
#         # X     = |n_users|x|n_items|     training data
#         # XtX   = |n_items|x|n_items|
#         # Z     = |n_users|x|n_feature_pairs|
#         # CCmask = |n_feature_pairs|x|n_items|
#         # ZtX    = |n_feature_pairs|x|n_items|
#         # PP    = |n_items|x|n_items|
#         # QQ    = |n_feature_pairs|x|n_feature_pairs|
#         # CC    = |n_feature_pairs|x|n_items|
#         # DD    = |n_feature_pairs|x|n_items|
#         # UU    = |n_feature_pairs|x|n_items|
#         # BB    = |n_items|x|n_items|
#
#
#
#         self._create_feature_pairs()
#
#         ### create the higher-order matrices
#         start_time = time.time()
#         self._print("Creating Higher-Order Matrices...")
#         ZtZ = self.Z.transpose() * self.Z
#         self.ZtX = self.Z.transpose() * self.X
#         ZtZdiag = deepcopy(ZtZ.diagonal())
#         self._print("Creating Higher-Order Matrices... done in {:.2f} {}.".format(*seconds_to_biggest_unit(time.time()-start_time)))
#
#         # precompute for BB
#         start_time = time.time()
#         self._print("Precomputing BB and CC...")
#         self.ii_diag = np.diag_indices(self.XtX.shape[0])
#         self.XtX[self.ii_diag] = self.XtXdiag + lambdaBB
#         self.PP=np.linalg.inv(self.XtX.todense())
#
#         # precompute for CC
#         self.ii_diag_ZZ=np.diag_indices(ZtZ.shape[0])
#         ZtZ[self.ii_diag_ZZ] = ZtZdiag+lambdaCC+rho
#         self.QQ=np.linalg.inv(ZtZ)
#
#         self._print("Precomputing BB and CC... done in {:.2f} {}.".format(*seconds_to_biggest_unit(time.time()-start_time)))
#
#         # initialize
#         self.CC = np.zeros( (ZtZ.shape[0], self.XtX.shape[0]),dtype=np.float )
#         self.DD = np.zeros( (ZtZ.shape[0], self.XtX.shape[0]),dtype=np.float )
#         self.UU = np.zeros( (ZtZ.shape[0], self.XtX.shape[0]),dtype=np.float ) # is Gamma in paper
#         self.BB = np.zeros( (self.URM_train.shape[0], self.URM_train.shape[0]),dtype=np.float )
#
#         ########################### Earlystopping
#
#         self._prepare_model_for_validation()
#         self._update_best_model()
#
#         self._train_with_early_stopping(epochs,
#                                         algorithm_name = self.RECOMMENDER_NAME,
#                                         **earlystopping_kwargs)
#
#         self.BB = self.BB_best
#         self.CC = self.CC_best
#
#
#     def _prepare_model_for_validation(self):
#         pass
#
#
#     def _update_best_model(self):
#         self.BB_best = self.BB.copy()
#         self.CC_best = self.CC.copy()
#
#
#     def _run_epoch(self, num_epoch):
#         #
#         # print("epoch {}".format(iter))
#
#         # learn BB
#         self.XtX[self.ii_diag] = self.XtXdiag
#         self.BB = self.PP.dot(self.XtX-self.ZtX.T.dot(self.CC))
#         gamma = np.diag(self.BB) / np.diag(self.PP)
#         self.BB -= self.PP *gamma
#
#         # learn CC
#         self.CC= self.QQ.dot(self.ZtX-self.ZtX.dot(self.BB) +self.rho *(self.DD-self.UU))
#
#         # learn DD
#         DD=  self.CC  * self.CCmask
#         #DD= np.maximum(0.0, DD) # if you want to enforce non-negative parameters
#
#         # learn UU (is Gamma in paper)
#         self.UU+= self.CC-DD
#
#
#
#     def save_model(self, folder_path, file_name = None):
#
#         if file_name is None:
#             file_name = self.RECOMMENDER_NAME
#
#         self._print("Saving model in file '{}'".format(folder_path + file_name))
#
#         data_dict_to_save = {"BB": self.BB,
#                              "CC": self.CC,
#                              "threshold": self.threshold,
#                             }
#
#         dataIO = DataIO(folder_path=folder_path)
#         dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)
#
#         self._print("Saving complete")
#
#
#     def load_model(self, folder_path, file_name = None):
#         super(NegHOSLIMRecommender, self).load_model(folder_path, file_name = file_name)
#
#         self._create_feature_pairs()
#




class NegHOSLIMRecommender(BaseItemSimilarityMatrixRecommender, Incremental_Training_Early_Stopping):
    """
    """

    RECOMMENDER_NAME = "NegHOSLIMRecommender"

    def __init__(self, URM_train, verbose = True):
        super(NegHOSLIMRecommender, self).__init__(URM_train, verbose = verbose)



    def _compute_item_score(self, user_id_array, items_to_compute = None):

        Xtest = self.URM_train[user_id_array,:]
        Ztest = self.Z[user_id_array,:]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
            BB_items_to_compute = self.BB[items_to_compute,:][:,items_to_compute]
            item_scores[:, items_to_compute] = (Xtest[:, items_to_compute]).dot(BB_items_to_compute) + \
                                               Ztest.dot(self.CC[:, items_to_compute])

        else:
            item_scores = (Xtest).dot(self.BB) + Ztest.dot(self.CC)

        return item_scores


    def _create_feature_pairs(self):

        self.X = self.URM_train

        start_time = time.time()
        self._print("Creating Feature Pairs...")

        self.XtX=np.array( ( self.X.transpose() * self.X).todense())
        self.XtXdiag=deepcopy(np.diag(self.XtX))

        ### create the list of feature-pairs and the higher-order matrix Z
        self.XtX[ np.diag_indices(self.XtX.shape[0]) ] = self.XtXdiag #if code is re-run, ensure that the diagonal is correct
        ii_feature_pairs = create_list_feature_pairs(self.XtX, self.threshold)

        # print("number of feature-pairs: {}".format(len(ii_feature_pairs[0])))
        self.Z, self.CCmask = create_matrix_Z(ii_feature_pairs, self.X)
        # Z_test_data_tr , _ = create_matrix_Z(ii_feature_pairs, test_data_tr)

        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print("Creating Feature Pairs... done in {:.2f} {}. Number of Feature-Pairs: {}".format( new_time_value, new_time_unit, len(ii_feature_pairs[0])))



    def fit(self, epochs=300, threshold = 3.5e4, lambdaBB = 5e2, lambdaCC = 5e3, rho = 1e5,
            **earlystopping_kwargs):

        self.rho = rho
        self.threshold = threshold

        ####### Data structures summary
        # X     = |n_users|x|n_items|     training data
        # XtX   = |n_items|x|n_items|
        # Z     = |n_users|x|n_feature_pairs|
        # CCmask = |n_feature_pairs|x|n_items|
        # ZtX    = |n_feature_pairs|x|n_items|
        # PP    = |n_items|x|n_items|
        # QQ    = |n_feature_pairs|x|n_feature_pairs|
        # CC    = |n_feature_pairs|x|n_items|
        # DD    = |n_feature_pairs|x|n_items|
        # UU    = |n_feature_pairs|x|n_items|
        # BB    = |n_items|x|n_items|



        self._create_feature_pairs()

        ### create the higher-order matrices
        start_time = time.time()
        self._print("Creating Higher-Order Matrices...")
        ZtZ=np.array(  (self.Z.transpose() * self.Z).todense())
        self.ZtX=np.array( (self.Z.transpose() * self.X).todense())
        ZtZdiag=deepcopy(np.diag(ZtZ))
        self._print("Creating Higher-Order Matrices... done in {:.2f} {}.".format(*seconds_to_biggest_unit(time.time()-start_time)))

        # precompute for BB
        start_time = time.time()
        self._print("Precomputing BB and CC...")
        self.ii_diag=np.diag_indices(self.XtX.shape[0])
        self.XtX[self.ii_diag] = self.XtXdiag+lambdaBB
        self.PP=np.linalg.inv(self.XtX)

        # precompute for CC
        self.ii_diag_ZZ=np.diag_indices(ZtZ.shape[0])
        ZtZ[self.ii_diag_ZZ] = ZtZdiag+lambdaCC+rho
        self.QQ=np.linalg.inv(ZtZ)

        self._print("Precomputing BB and CC... done in {:.2f} {}.".format(*seconds_to_biggest_unit(time.time()-start_time)))

        # initialize
        self.CC = np.zeros( (ZtZ.shape[0], self.XtX.shape[0]),dtype=np.float )
        self.DD = np.zeros( (ZtZ.shape[0], self.XtX.shape[0]),dtype=np.float )
        self.UU = np.zeros( (ZtZ.shape[0], self.XtX.shape[0]),dtype=np.float ) # is Gamma in paper
        self.BB = np.zeros( (self.URM_train.shape[0], self.URM_train.shape[0]),dtype=np.float )

        ########################### Earlystopping

        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.BB = self.BB_best
        self.CC = self.CC_best


    def _prepare_model_for_validation(self):
        pass


    def _update_best_model(self):
        self.BB_best = self.BB.copy()
        self.CC_best = self.CC.copy()


    def _run_epoch(self, num_epoch):
        #
        # print("epoch {}".format(iter))

        # learn BB
        self.XtX[self.ii_diag] = self.XtXdiag
        self.BB = self.PP.dot(self.XtX-self.ZtX.T.dot(self.CC))
        gamma = np.diag(self.BB) / np.diag(self.PP)
        self.BB -= self.PP *gamma

        # learn CC
        self.CC= self.QQ.dot(self.ZtX-self.ZtX.dot(self.BB) +self.rho *(self.DD-self.UU))

        # learn DD
        DD=  self.CC  * self.CCmask
        #DD= np.maximum(0.0, DD) # if you want to enforce non-negative parameters

        # learn UU (is Gamma in paper)
        self.UU+= self.CC-DD



    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"BB": self.BB,
                             "CC": self.CC,
                             "threshold": self.threshold,
                            }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")


    def load_model(self, folder_path, file_name = None):
        super(NegHOSLIMRecommender, self).load_model(folder_path, file_name = file_name)

        self._create_feature_pairs()
