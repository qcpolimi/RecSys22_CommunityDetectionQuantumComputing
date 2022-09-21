#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/19
@author: Anonymous
"""


######################################################################
##########                                                  ##########
##########                  NON PERSONALIZED                ##########
##########                                                  ##########
######################################################################
from recsys.Recommenders.NonPersonalizedRecommender import TopPop, Random, GlobalEffects



######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
from recsys.Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from recsys.Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from recsys.Recommenders.SLIM.NegHOSLIM import NegHOSLIMRecommender
from recsys.Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from recsys.Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_SLIMElasticNetRecommender
from recsys.Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from recsys.Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from recsys.Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from recsys.Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from recsys.Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from recsys.Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from recsys.Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from recsys.Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
from recsys.Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask as MultVAERecommender


######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
from recsys.Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from recsys.Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender



######################################################################
##########                                                  ##########
##########                       HYBRID                     ##########
##########                                                  ##########
######################################################################
from recsys.Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from recsys.Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender
from recsys.Recommenders.FactorizationMachines.LightFMRecommender import LightFMUserHybridRecommender, LightFMItemHybridRecommender