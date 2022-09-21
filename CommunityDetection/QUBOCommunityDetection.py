import time

import dimod
import numpy as np
from dimod import SampleSet

from CommunityDetection.BaseCommunityDetection import BaseCommunityDetection
from utils.DataIO import DataIO
from utils.sampling import sample_wrapper


class QUBOCommunityDetection(BaseCommunityDetection):
    is_qubo = True
    name = 'QUBOCommunityDetection'

    def __init__(self, urm):
        super(QUBOCommunityDetection, self).__init__(urm=urm)

        self._Q = None

    def save_model(self, folder_path, file_name):
        data_dict_to_save = {
            '_Q': self._Q,
            '_fit_time': self._fit_time,
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

    def run(self, sampler, sampler_args=None) -> [SampleSet, dict, float]:
        if sampler_args is None:
            sampler_args = {}
        BQM = dimod.BinaryQuadraticModel(self._Q, vartype=dimod.BINARY)

        start_time = time.time()
        sampleset, sampler_info = sample_wrapper(BQM, sampler, **sampler_args)
        run_time = time.time() - start_time

        return sampleset, sampler_info, run_time

    def get_Q_adjacency(self):
        BQM = dimod.BinaryQuadraticModel(self._Q, vartype=dimod.BINARY)
        return dimod.to_networkx_graph(BQM)

    @staticmethod
    def get_comm_from_sample(sample, n, **kwargs):
        n_features = len(sample)
        comm = np.zeros(n_features, dtype=int)
        for k, v in sample.items():
            if v == 1:
                ind = int(k)
                comm[ind] = 1

        return comm[:n], comm[n:]
