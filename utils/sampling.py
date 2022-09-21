import neal
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from hybrid import HybridSampler


def __json_convert_not_serializable(o):
    """
    Json cannot serialize automatically some data types, for example numpy integers (int32).
    This may be a limitation of numpy-json interfaces for Python 3.6 and may not occur in Python 3.7
    :param o:
    :return:
    """

    if isinstance(o, np.integer):
        return int(o)

    if isinstance(o, np.bool_):
        return bool(o)

    return o


def sample_wrapper(BQM, sampler, **sampler_args):
    if isinstance(sampler, EmbeddingComposite):
        sampler_args['return_embedding'] = True

        # Hyperparameters of type int32, int64, bool_ are not json serializable
        # transform them from numpy to native Python types
        for key, value in sampler_args.items():
            sampler_args[key] = __json_convert_not_serializable(value)

    if isinstance(sampler, LeapHybridSampler):
        sampler_args.pop('num_reads')

    sampleset = sampler.sample(BQM, **sampler_args)
    # sampleset_df = sampleset.aggregate().to_pandas_dataframe()
    sampler_info = sampleset.info

    # Change the format of the infos returned by the QPU sampler
    if isinstance(sampler, EmbeddingComposite):

        for key, value in sampler_info['timing'].items():
            sampler_info['timing_' + key] = value

        del sampler_info['timing']

        for key, value in sampler_info['embedding_context'].items():
            sampler_info[key] = value

        del sampler_info['embedding_context']

        for key, value in sampler_info.items():
            if 'embedding' in key:
                sampler_info[key] = str(value)

    elif isinstance(sampler, DWaveSampler):

        for key, value in sampler_info['timing'].items():
            sampler_info['timing_' + key] = value

        del sampler_info['timing']

    elif isinstance(sampler, LeapHybridSampler):
        pass

    elif isinstance(sampler, HybridSampler):
        best_state = sampler_info.pop('best_state', None)
        if best_state is not None:
            sampleset = best_state.samples
            sampler_info = sampleset.info

    elif isinstance(sampler, neal.SimulatedAnnealingSampler):
        sampler_info['beta_range_min'] = sampler_info['beta_range'][0]
        sampler_info['beta_range_max'] = sampler_info['beta_range'][1]
        del sampler_info['beta_range']

    # return sampleset_df, sampler_info
    return sampleset, sampler_info
