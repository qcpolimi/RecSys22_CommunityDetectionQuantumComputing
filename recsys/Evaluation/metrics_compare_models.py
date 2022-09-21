#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Anonymous
"""


import numpy as np
from recsys.Evaluation.metrics import _Metrics_Object


####################################################################################################################
###############                 ACCURACY METRICS
####################################################################################################################


class CommonItems(_Metrics_Object):
    """
    CommonItems, defined as the number of items present in both lists
    """

    def __init__(self, cutoff):
        super(CommonItems, self).__init__()
        self.counter = 0
        self.n_users = 0
        self.cutoff = cutoff

    def add_recommendations(self, common_items_list):
        self.counter += len(common_items_list)
        self.n_users += 1

    def get_metric_value(self):
        return self.counter/(self.n_users * self.cutoff)





class CommonHits(_Metrics_Object):
    """
    CommonHits, defined as the number of common hits present in both boolean arrays
    """

    def __init__(self):
        super(CommonHits, self).__init__()
        self.counter = 0
        self.n_users = 0

    def add_recommendations(self, common_is_relevant):
        self.counter += np.sum(common_is_relevant)
        self.n_users += 1

    def get_metric_value(self):
        return self.counter/self.n_users




class CommonHitsQuota(_Metrics_Object):
    """
    CommonHitsQuota, defined as percentage of common items over the total set of hits
    The size of the total set of hits is |A + B| = |A| + |B| - |A intersect B|
    """

    def __init__(self):
        super(CommonHitsQuota, self).__init__()
        self.cumulative_quota = 0.0
        self.n_users = 0

    def add_recommendations(self, is_relevant_1, is_relevant_2, common_is_relevant):
        
        total_hits = np.sum(is_relevant_1) + np.sum(is_relevant_2) - np.sum(common_is_relevant)
        
        if total_hits != 0:
            self.cumulative_quota += np.sum(common_is_relevant)/total_hits
            self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_quota/self.n_users