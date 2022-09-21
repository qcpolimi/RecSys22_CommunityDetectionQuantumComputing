import time

import numpy as np

from CommunityDetection.QUBOCommunityDetection import QUBOCommunityDetection


class QUBOBipartiteProjectedCommunityDetection(QUBOCommunityDetection):
    filter_items = False
    name = 'QUBOBipartiteProjectedCommunityDetection'

    def __init__(self, urm):
        super(QUBOBipartiteProjectedCommunityDetection, self).__init__(urm=urm)

    def fit(self, weighted=True, threshold=None):
        start_time = time.time()

        W = self.urm * self.urm.T

        if not weighted:
            W = (W > 0) * 1

        W.setdiag(0)
        W.eliminate_zeros()

        k = W.sum(axis=1)
        m = k.sum() // 2

        P = k @ k.T / (2 * m)

        B = W - P

        if threshold is not None:
            B[np.abs(B) < threshold] = 0

        self._Q = -B / m  # Normalized QUBO matrix

        self._fit_time = time.time() - start_time

    @staticmethod
    def get_comm_from_sample(sample, n_users, n_items=0):
        users, _ = super(QUBOBipartiteProjectedCommunityDetection,
                         QUBOBipartiteProjectedCommunityDetection).get_comm_from_sample(sample, n_users)
        return users, np.zeros(n_items)
