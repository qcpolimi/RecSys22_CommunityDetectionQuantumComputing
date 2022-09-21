"""
Created on 27/09/2019

@author: Anonymous
"""


from recsys.Recommenders.Recommender_import_list import *

class RecommenderFactory(object):
    """RecommenderFactory"""

    def __init__(self,  URM = None,
                        ICM = None,
                        UCM = None,
                 ):

        super(RecommenderFactory, self).__init__()

        assert URM is not None, "RecommenderFactory: Model requires URM, which is None."

        self._URM = URM
        self._ICM = ICM
        self._UCM = UCM


    def get_recommender_instance(self, recommender_class):

        constructor_args_dict = {"URM": self._URM}

        if recommender_class is ItemKNNCBFRecommender:
            assert self._ICM is not None, "RecommenderFactory: Model requires ICM, which is None."
            constructor_args_dict["ICM"] = self._ICM

        elif recommender_class is UserKNNCBFRecommender:
            assert self._UCM is not None, "RecommenderFactory: Model requires UCM, which is None."
            constructor_args_dict["UCM"] = self._UCM

        # Allow to add a custom function?

        recommender_instance = recommender_class(**constructor_args_dict)

        return recommender_instance
