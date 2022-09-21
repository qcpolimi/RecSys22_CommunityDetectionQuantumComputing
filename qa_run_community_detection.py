import shutil
import time

import dimod
import minorminer
import numpy as np
from dwave.system import DWaveSampler, LeapHybridSampler, FixedEmbeddingComposite
from greedy import SteepestDescentSampler
from neal import SimulatedAnnealingSampler
from tabu import TabuSampler

from CommunityDetection import Communities, Community, BaseCommunityDetection, QUBOCommunityDetection, \
    QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection, EmptyCommunityError, \
    get_community_folder_path
from run_community_detection import check_communities
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader
from utils.DataIO import DataIO
from utils.types import Iterable, Type
from utils.urm import load_data, merge_sparse_matrices, get_community_urm


def load_communities(folder_path, method, sampler: Type[dimod.Sampler] = None, n_iter=0, n_comm=None):
    method_folder_path = f'{folder_path}{method.name}/'
    folder_suffix = '' if sampler is None else f'{sampler.__name__}/'

    try:
        communities = Communities.load(method_folder_path, 'communities', n_iter=n_iter, n_comm=n_comm,
                                       folder_suffix=folder_suffix)
        print(f'Loaded previously computed communities for {communities.num_iters + 1} iterations.')
    except FileNotFoundError:
        print('No communities found to load.')
        communities = None
    return communities


def main(data_reader_classes, method_list: Iterable[Type[BaseCommunityDetection]],
         base_sampler_list: Iterable[Type[dimod.Sampler]], sampler_list: Iterable[dimod.Sampler],
         result_folder_path: str, num_iters: int = 3):
    split_quota = [80, 10, 10]
    user_wise = True
    make_implicit = True
    threshold = None

    fit_args = {
        'threshold': None,
    }

    sampler_args = {
        'num_reads': 100,
    }

    save_model = True

    for data_reader_class in data_reader_classes:
        data_reader = data_reader_class()
        dataset_name = data_reader._get_dataset_name()
        dataset_folder_path = f'{result_folder_path}{dataset_name}/'

        urm_train, urm_validation, urm_test = load_data(data_reader, split_quota=split_quota, user_wise=user_wise,
                                                        make_implicit=make_implicit, threshold=threshold)
        urm_train_last_test = merge_sparse_matrices(urm_train, urm_validation)

        for method in method_list:
            qa_cd_per_method(urm_train_last_test, method, base_sampler_list, sampler_list, dataset_folder_path,
                             num_iters=num_iters, fit_args=fit_args, sampler_args=sampler_args, save_model=save_model)


def qa_cd_per_method(cd_urm, method, base_sampler_list, sampler_list, folder_path, num_iters=1, **kwargs):
    if method.is_qubo:
        for base_sampler in base_sampler_list:
            for sampler in sampler_list:
                qa_community_detection(cd_urm, method, folder_path, base_sampler=base_sampler, sampler=sampler,
                                       num_iters=num_iters, **kwargs)
    else:
        for sampler in sampler_list:
            qa_community_detection(cd_urm, method, folder_path, sampler=sampler, num_iters=num_iters, **kwargs)


def qa_community_detection(cd_urm, method, folder_path, base_sampler: Type[dimod.Sampler] = None,
                           sampler: dimod.Sampler = None, num_iters: int = 1, **kwargs):
    communities: Communities = load_communities(folder_path, method, base_sampler)
    if communities is None:
        print(f'Could not load a community for {folder_path}, {method.name}, {base_sampler.__name__}')
        return

    starting_iter = None
    for n_iter in range(num_iters):
        if check_communities_size(communities, n_iter, method.filter_users, method.filter_items):
            starting_iter = n_iter
            if method is QUBOBipartiteProjectedCommunityDetection:
                starting_iter += 1
            break

    if starting_iter is None:
        print(f'Could not find a suitable iteration in range of {num_iters} iterations.')
        return

    original_iters = communities.num_iters
    communities.reset_from_iter(starting_iter)

    for n_iter in range(starting_iter, num_iters):
        try:
            communities = qa_cd_per_iter(cd_urm, method, folder_path, base_sampler=base_sampler, sampler=sampler,
                                         communities=communities, n_iter=n_iter, **kwargs)
        except EmptyCommunityError as e:
            print(e)
            print(f'Stopping at iteration {n_iter}.')
            clean_empty_iteration(n_iter, folder_path, method, base_sampler=base_sampler, sampler=sampler)
            break
        except ValueError as e:
            print(e)
            print(f'Could not find an embedding at iteration {n_iter}.')
            clean_empty_iteration(n_iter, folder_path, method, base_sampler=base_sampler, sampler=sampler)
            if n_iter > original_iters:
                break
            continue


def qa_cd_per_iter(cd_urm, method, folder_path, base_sampler: Type[dimod.Sampler] = None, sampler: dimod.Sampler = None,
                   communities: Communities = None,
                   n_iter: int = 0, **kwargs):
    print(f'Running community detection iteration {n_iter} with {method.name}...')
    if communities is None:
        assert n_iter == 0, 'If no communities are given this must be the first iteration.'

        communities = qa_run_cd(cd_urm, method, folder_path, base_sampler=base_sampler, sampler=sampler, n_iter=n_iter,
                                n_comm=None, **kwargs)
    else:
        assert n_iter != 0, 'Cannot be the first iteration if previously computed communities are given.'

        new_communities = []
        n_comm = 0
        for community in communities.iter(n_iter):
            cd = qa_run_cd(cd_urm, method, folder_path, base_sampler=base_sampler, sampler=sampler, community=community,
                           n_iter=n_iter, n_comm=n_comm, **kwargs)
            new_communities.append(cd)
            n_comm += 1
        communities.add_iteration(new_communities)

    print('Saving community detection results...')
    method_folder_path = f'{folder_path}{method.name}/'
    folder_suffix = get_folder_suffix(base_sampler, sampler)

    communities.save_from_iter(n_iter, method_folder_path, 'communities', folder_suffix=folder_suffix)

    return communities


def qa_run_cd(cd_urm, method: Type[BaseCommunityDetection], folder_path: str, base_sampler: Type[dimod.Sampler] = None,
              sampler: dimod.Sampler = None, community: Community = None, n_iter: int = 0, n_comm: int = None,
              **kwargs) -> Communities:
    n_users, n_items = cd_urm.shape
    user_index = np.arange(n_users)
    item_index = np.arange(n_items)

    if community is not None:
        cd_urm, user_index, item_index = get_community_urm(cd_urm, community, filter_users=method.filter_users,
                                                           filter_items=method.filter_items, remove=True)
    n_users, n_items = cd_urm.shape

    m: BaseCommunityDetection = method(cd_urm)

    method_folder_path = f'{folder_path}{m.name}/'
    folder_suffix = get_folder_suffix(base_sampler, sampler)
    method_folder_path = get_community_folder_path(method_folder_path, n_iter=n_iter, folder_suffix=folder_suffix)

    comm_file_suffix = f'{n_comm:02d}' if n_comm is not None else ''
    model_file_name = f'model{comm_file_suffix}'

    try:
        m.load_model(method_folder_path, model_file_name)
        print('Loaded previously computed CD model.')
    except FileNotFoundError:
        fit_args = kwargs.get('fit_args', {})
        m.fit(**fit_args)

        if kwargs.get('save_model', True):
            print('Saving CD model...')
            m.save_model(method_folder_path, model_file_name)

    dataIO = DataIO(method_folder_path)
    run_file_name = f'run{comm_file_suffix}'

    try:
        run_dict = dataIO.load_data(run_file_name)
        if sampler is not None:
            assert method.is_qubo, 'Cannot use a QUBO sampler on a non-QUBO method.'
            m: QUBOCommunityDetection
            sampleset = dimod.SampleSet.from_serializable(run_dict['sampleset'])
            users, items = m.get_comm_from_sample(sampleset.first.sample, n_users, n_items=n_items)
        else:
            users = run_dict['users']
            items = run_dict['items']
        print(f'Loaded previous CD run {n_comm:02d}.')

    except FileNotFoundError:
        print('Running CD...')
        if sampler is not None:
            assert method.is_qubo, 'Cannot use a QUBO sampler on a non-QUBO method.'
            m: QUBOCommunityDetection

            embedding = {}
            embedding_time = 0
            if isinstance(sampler, DWaveSampler):
                embedding_time = time.time()
                embedding = minorminer.find_embedding(m.get_Q_adjacency(), sampler.edgelist)
                embedding_time = time.time() - embedding_time

                sampler = FixedEmbeddingComposite(sampler, embedding)

            sampler_args = kwargs.get('sampler_args', {})
            sampleset, sampler_info, run_time = m.run(sampler, sampler_args)

            data_dict_to_save = {
                'sampleset': sampleset.to_serializable(),
                'sampler_info': sampler_info,
                'run_time': run_time,
            }

            if isinstance(sampler, FixedEmbeddingComposite):
                data_dict_to_save['embedding'] = embedding
                data_dict_to_save['embedding_time'] = embedding_time

            users, items = m.get_comm_from_sample(sampleset.first.sample, n_users, n_items=n_items)
        else:
            users, items, run_time = m.run()

            data_dict_to_save = {
                'users': users,
                'items': items,
                'run_time': run_time,
            }

        dataIO.save_data(run_file_name, data_dict_to_save)

    communities = Communities(users, items, user_index, item_index)
    check_communities(communities, m.filter_users, m.filter_items)
    return communities


def get_folder_suffix(base_sampler, sampler):
    folder_suffix = f'{sampler.__class__.__name__}/'
    if base_sampler is not None:
        folder_suffix = f'{base_sampler.__name__}_{folder_suffix}'
    return folder_suffix


def check_communities_size(communities: Communities, n_iter: int, check_users, check_items):
    if communities is None:
        print('Non-existing communities. Please use non-null ones.')
        return False

    if communities.num_iters < n_iter:
        return False

    for community in communities.iter(n_iter):
        n_nodes = len(community.users) if check_users else 0
        n_nodes += len(community.items) if check_items else 0
        if n_nodes > 170:
            print(f'Communities size exceeded maximum size for the Quantum Annealer at iteration {n_iter}.')
            return False
    return True


def clean_empty_iteration(n_iter: int, folder_path: str, method: Type[BaseCommunityDetection],
                          base_sampler: Type[dimod.Sampler], sampler: dimod.Sampler = None, start_iter: int = 0):
    folder_suffix = get_folder_suffix(base_sampler, sampler)
    folder_path = f'{folder_path}{method.name}/'
    rm_folder_path = get_community_folder_path(folder_path, n_iter=n_iter, folder_suffix=folder_suffix)
    shutil.rmtree(rm_folder_path)

    base_folder_suffix = '' if base_sampler is None else f'{base_sampler.__class__.__name__}/'
    try:
        communities = Communities.load(folder_path, 'communities', n_iter=0, folder_suffix=base_folder_suffix)

        communities.load_from_iter(start_iter, folder_path, 'communities', folder_suffix=folder_suffix)
        print(f'Reloaded previously computed communities for {communities.num_iters + 1} iterations.')

        communities.save_from_iter(start_iter, folder_path, 'communities', folder_suffix=folder_suffix)
        print('Saved the cleaned communities.')
    except FileNotFoundError:
        print('Cannot load communities, cleaning not complete.')


if __name__ == '__main__':
    data_reader_classes = [Movielens100KReader, Movielens1MReader, FilmTrustReader, MovielensHetrec2011Reader,
                           LastFMHetrec2011Reader, FrappeReader, CiteULike_aReader, CiteULike_tReader]
    method_list = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection]
    base_sampler_list = [LeapHybridSampler, SimulatedAnnealingSampler, SteepestDescentSampler, TabuSampler]
    sampler_list = [DWaveSampler(solver={'topology__type': 'pegasus'})]
    num_iters = 15
    result_folder_path = './results/'
    main(data_reader_classes, method_list, base_sampler_list, sampler_list, result_folder_path, num_iters=num_iters)
