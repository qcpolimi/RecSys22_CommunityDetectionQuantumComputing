import os.path
import os.path
import time

import dimod
import greedy
import neal
import numpy as np
import tabu
from dwave.system import LeapHybridSampler, DWaveSampler

from CommunityDetection import BaseCommunityDetection, QUBOBipartiteCommunityDetection, \
    QUBOBipartiteProjectedCommunityDetection, Communities, CommunityDetectionRecommender, \
    get_community_folder_path
from qa_run_community_detection import qa_run_cd
from recsys.Data_manager import Movielens100KReader, Movielens1MReader, FilmTrustReader, FrappeReader, \
    MovielensHetrec2011Reader, LastFMHetrec2011Reader, CiteULike_aReader, CiteULike_tReader
from recsys.Evaluation.Evaluator import EvaluatorHoldout
from recsys.Recommenders.BaseRecommender import BaseRecommender
from recsys.Recommenders.NonPersonalizedRecommender import TopPop
from utils.DataIO import DataIO
from utils.types import Iterable, Type
from utils.urm import get_community_urm, load_data, merge_sparse_matrices

CUTOFF_LIST = [5, 10, 20, 30, 40, 50, 100]


def load_communities(folder_path, method, sampler=None, n_iter=0, n_comm=None):
    method_folder_path = f'{folder_path}{method.name}/'
    folder_suffix = '' if sampler is None else f'{sampler.__class__.__name__}/'

    try:
        communities = Communities.load(method_folder_path, 'communities', n_iter=n_iter, n_comm=n_comm,
                                       folder_suffix=folder_suffix)
        print(f'Loaded previously computed communities for {communities.num_iters + 1} iterations.')
    except FileNotFoundError:
        print('No communities found to load. Computing new communities...')
        communities = None
    return communities


def train_all_data_recommender(recommender: Type[BaseRecommender], urm_train_last_test, urm_test, dataset_name: str,
                               results_folder_path: str):
    recommender_name = recommender.RECOMMENDER_NAME
    output_folder_path = f'{results_folder_path}{dataset_name}/{recommender_name}/'

    evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=CUTOFF_LIST)

    print(f'Training {recommender_name} on all data...')

    time_on_train = time.time()
    rec = recommender(urm_train_last_test)
    rec.fit()
    time_on_train = time.time() - time_on_train

    time_on_test = time.time()
    result_df, result_string = evaluator_test.evaluateRecommender(rec)
    time_on_test = time.time() - time_on_test

    data_dict_to_save = {
        'result_df': result_df,
        'result_string': result_string,
        'time_on_train': time_on_train,
        'time_on_test': time_on_test,
    }

    output_dataIO = DataIO(output_folder_path)
    output_dataIO.save_data('baseline', data_dict_to_save)

    rec.save_model(output_folder_path, f'{recommender_name}_best_model_last')


def train_recommender_on_community(recommender, community, urm_train, urm_validation, urm_test, dataset_name,
                                   results_folder_path, method_folder_path, n_iter=0, n_comm=None, folder_suffix='',
                                   **kwargs):
    recommender_name = recommender.RECOMMENDER_NAME
    print(f'Training {recommender_name} on community {n_comm if n_comm is not None else ""} of iteration {n_iter}...')

    output_folder_path = get_community_folder_path(method_folder_path, n_iter=n_iter, n_comm=n_comm,
                                                   folder_suffix=folder_suffix)
    output_folder_path = f'{output_folder_path}{recommender_name}/'

    base_recommender_path = f'{results_folder_path}{dataset_name}/{recommender_name}/'

    c_urm_train, _, _ = get_community_urm(urm_train, community=community, filter_items=False)
    c_urm_validation, _, _ = get_community_urm(urm_validation, community=community, filter_items=False)
    c_urm_test, _, _ = get_community_urm(urm_test, community=community, filter_items=False)
    c_urm_train_last_test = merge_sparse_matrices(c_urm_train, c_urm_validation)

    ignore_users = np.arange(c_urm_train_last_test.shape[0])[np.logical_not(community.user_mask)]
    evaluator_validation = EvaluatorHoldout(c_urm_validation, cutoff_list=CUTOFF_LIST, ignore_users=ignore_users)
    evaluator_test = EvaluatorHoldout(c_urm_test, cutoff_list=CUTOFF_LIST, ignore_users=ignore_users)

    time_on_train = time.time()
    validation_recommender = recommender(c_urm_train)
    validation_recommender.fit()
    time_on_train = time.time() - time_on_train

    time_on_validation = time.time()
    result_df, result_string = evaluator_validation.evaluateRecommender(validation_recommender)
    time_on_validation = time.time() - time_on_validation

    data_dict_to_save = {
        'result_df': result_df,
        'result_string': result_string,
        'time_on_train': time_on_train,
        'time_on_validation': time_on_validation,
    }

    output_dataIO = DataIO(output_folder_path)
    output_dataIO.save_data('validation', data_dict_to_save)

    time_on_train = time.time()
    comm_recommender = recommender(c_urm_train_last_test)
    comm_recommender.fit()
    time_on_train = time.time() - time_on_train

    time_on_test = time.time()
    result_df, result_string = evaluator_test.evaluateRecommender(comm_recommender)
    time_on_test = time.time() - time_on_test

    data_dict_to_save = {
        'result_df': result_df,
        'result_string': result_string,
        'time_on_train': time_on_train,
        'time_on_test': time_on_test,
    }

    output_dataIO = DataIO(output_folder_path)
    output_dataIO.save_data('test', data_dict_to_save)

    print(f'Evaluating base model on community {n_comm if n_comm is not None else ""} of iteration {n_iter}...')
    base_recommender = recommender(c_urm_train_last_test)
    recommender_file_name = f'{recommender_name}_best_model_last'
    base_recommender.load_model(base_recommender_path, recommender_file_name)
    base_evaluator_test = EvaluatorHoldout(c_urm_test, cutoff_list=CUTOFF_LIST, ignore_users=ignore_users)

    time_on_test = time.time()
    result_df, result_string = base_evaluator_test.evaluateRecommender(base_recommender)
    time_on_test = time.time() - time_on_test

    baseline_dict = {
        'result_df': result_df,
        'result_string': result_string,
        'time_on_test': time_on_test,
    }
    output_dataIO.save_data('baseline', baseline_dict)
    print(result_string)

    return comm_recommender


def evaluate_recommender(urm_train_last_test, urm_test, communities, recommenders, output_folder_path, recommender_name,
                         n_iter=None):
    print(f'Evaluating {recommender_name} on the result of community detection.')

    recommender = CommunityDetectionRecommender(urm_train_last_test, communities=communities, recommenders=recommenders,
                                                n_iter=n_iter)

    evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=CUTOFF_LIST)
    time_on_test = time.time()
    result_df, result_string = evaluator_test.evaluateRecommender(recommender)
    time_on_test = time.time() - time_on_test

    result_dict = {
        'result_df': result_df,
        'result_string': result_string,
        'time_on_test': time_on_test,
    }

    dataIO = DataIO(output_folder_path)
    dataIO.save_data(f'cd_{recommender_name}', result_dict)
    print(result_string)


def main(data_reader_classes, method_list: Iterable[Type[BaseCommunityDetection]],
         sampler_list: Iterable[dimod.Sampler], recommender_list: Iterable[Type[BaseRecommender]],
         result_folder_path: str):
    split_quota = [80, 10, 10]
    user_wise = False
    make_implicit = False
    threshold = None

    recsys_args = {
        'cutoff_to_optimize': 10,
        'cutoff_list': CUTOFF_LIST,
        'n_cases': 50,
        'n_random_starts': 15,
        'metric_to_optimize': 'NDCG',
        'resume_from_saved': True,
        'similarity_type_list': ['cosine'],
    }

    save_model = True

    for data_reader_class in data_reader_classes:
        data_reader = data_reader_class()
        dataset_name = data_reader._get_dataset_name()
        urm_train, urm_validation, urm_test = load_data(data_reader, split_quota=split_quota, user_wise=user_wise,
                                                        make_implicit=make_implicit, threshold=threshold)

        urm_train_last_test = merge_sparse_matrices(urm_train, urm_validation)

        for recommender in recommender_list:
            recommender_name = recommender.RECOMMENDER_NAME
            output_folder_path = f'{result_folder_path}{dataset_name}/{recommender_name}/'
            if not os.path.exists(f'{output_folder_path}baseline.zip') or not os.path.exists(
                    f'{output_folder_path}{recommender_name}_best_model_last.zip'):
                train_all_data_recommender(recommender, urm_train_last_test, urm_test, dataset_name, result_folder_path)
            else:
                print(f'{recommender_name} already trained and evaluated on {dataset_name}.')

        for method in method_list:
            recommend_per_method(urm_train, urm_validation, urm_test, urm_train_last_test, method, sampler_list,
                                 recommender_list, dataset_name, result_folder_path, recsys_args=recsys_args.copy(),
                                 save_model=save_model)


def recommend_per_method(urm_train, urm_validation, urm_test, cd_urm, method, sampler_list, recommender_list,
                         dataset_name, folder_path, **kwargs):
    if method.is_qubo:
        for sampler in sampler_list:
            cd_recommendation(urm_train, urm_validation, urm_test, cd_urm, method, recommender_list, dataset_name,
                              folder_path, sampler=sampler, **kwargs)
    else:
        cd_recommendation(urm_train, urm_validation, urm_test, cd_urm, method, recommender_list, dataset_name,
                          folder_path, **kwargs)


def cd_recommendation(urm_train, urm_validation, urm_test, cd_urm, method, recommender_list, dataset_name, folder_path,
                      sampler: dimod.Sampler = None, **kwargs):
    dataset_folder_path = f'{folder_path}{dataset_name}/'
    communities = load_communities(dataset_folder_path, method, sampler)
    if communities is None:
        print(f'Could not load communitites for {dataset_folder_path}, {method}, {sampler}.')
        return

    method_path = f'{dataset_folder_path}{method.name}/'
    num_iters = communities.num_iters + 1
    starting_iter = None
    sampler_name = sampler.__class__.__name__
    for n_iter in range(num_iters):
        if os.path.exists(f'{method_path}iter{n_iter:02d}/{sampler_name}_DWaveSampler/'):
            print(f'Found QPU CD at iteration {n_iter}.')
            starting_iter = n_iter
            break

    if starting_iter is None:
        print(f'No QPU experiments for {dataset_name} with {method} + {sampler_name}')
        return

    communities.reset_from_iter(starting_iter)

    for n_iter in range(starting_iter, num_iters):
        new_communities = []
        n_comm = 0
        for community in communities.iter(n_iter):
            cd = qa_run_cd(cd_urm, method, dataset_folder_path, base_sampler=sampler.__class__, sampler=DWaveSampler(),
                           community=community, n_iter=n_iter, n_comm=n_comm, **kwargs)
            new_communities.append(cd)
            n_comm += 1
        communities.add_iteration(new_communities)

        recommend_per_iter(urm_train, urm_validation, urm_test, cd_urm, method, recommender_list, dataset_name,
                           folder_path, sampler=sampler, communities=communities, n_iter=n_iter, **kwargs)


def recommend_per_iter(urm_train, urm_validation, urm_test, cd_urm, method, recommender_list, dataset_name, folder_path,
                       sampler: dimod.Sampler = None, communities: Communities = None, n_iter: int = 0, **kwargs):
    method_folder_path = f'{folder_path}{dataset_name}/{method.name}/'
    folder_suffix = '' if sampler is None else f'{sampler.__class__.__name__}_DWaveSampler/'

    print(f'///Training recommenders for iteration {n_iter} on {dataset_name} with {method.name} and {folder_suffix}//')

    output_folder_path = get_community_folder_path(method_folder_path, n_iter=n_iter, folder_suffix=folder_suffix)
    for recommender in recommender_list:
        recommender_name = recommender.RECOMMENDER_NAME
        print(f'{recommender_name}...')

        if not os.path.exists(f'{output_folder_path}cd_{recommender_name}.zip'):
            n_comm = 0
            cd_recommenders = []
            for community in communities.iter(n_iter):
                comm_recommender = train_recommender_on_community(recommender, community, urm_train, urm_validation,
                                                                  urm_test, dataset_name, folder_path,
                                                                  method_folder_path, n_iter=n_iter, n_comm=n_comm,
                                                                  folder_suffix=folder_suffix)
                cd_recommenders.append(comm_recommender)
                n_comm += 1
            evaluate_recommender(cd_urm, urm_test, communities, cd_recommenders, output_folder_path,
                                 recommender_name, n_iter=n_iter)
        else:
            print('Recommender already trained and evaluated.')


if __name__ == '__main__':
    data_reader_classes = [Movielens100KReader, Movielens1MReader, FilmTrustReader, MovielensHetrec2011Reader,
                           LastFMHetrec2011Reader, FrappeReader, CiteULike_aReader, CiteULike_tReader]
    recommender_list = [TopPop]
    method_list = [QUBOBipartiteCommunityDetection, QUBOBipartiteProjectedCommunityDetection]
    sampler_list = [LeapHybridSampler(), neal.SimulatedAnnealingSampler(), greedy.SteepestDescentSampler(),
                    tabu.TabuSampler()]
    result_folder_path = './results/'
    main(data_reader_classes, method_list, sampler_list, recommender_list, result_folder_path)
