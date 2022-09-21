import numpy as np

from CommunityDetection.Communities import Communities
from recsys.Recommenders.BaseRecommender import BaseRecommender
from utils.types import List, NDArray


class CommunityDetectionRecommender(BaseRecommender):
    RECOMMENDER_NAME = "CommunityDetectionRecommender"

    def __init__(self, URM_train, communities: Communities, recommenders: List[BaseRecommender], n_iter=None,
                 verbose=True):
        super(CommunityDetectionRecommender, self).__init__(URM_train, verbose=verbose)

        self.communities = communities
        self.recommenders: List[BaseRecommender] = recommenders
        self.n_iter: int = n_iter if n_iter is not None else self.communities.num_iters

        assert len(self.recommenders) == 2 ** (self.n_iter + 1), \
            'Cannot use a different number of recommenders and communities.'

    def fit(self, n_iter=None, recommenders=None):
        if n_iter is not None:
            self.n_iter = n_iter
        if recommenders is not None:
            self.recommenders = recommenders
        assert len(self.recommenders) == 2 ** (self.n_iter + 1), \
            'Cannot use a different number of recommenders and communities.'

    def _compute_item_score(self, user_id_array: NDArray, items_to_compute=None):
        n_comm = 0
        item_scores = -np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32) * np.inf

        for community in self.communities.iter(self.n_iter):
            comm_users = community.users
            user_mask = np.isin(user_id_array, comm_users, assume_unique=True)
            users: NDArray = user_id_array[user_mask]

            if user_mask.any():
                recommender = self.recommenders[n_comm]
                item_scores[user_mask] = recommender._compute_item_score(users, items_to_compute=items_to_compute)

            n_comm += 1

        return item_scores
