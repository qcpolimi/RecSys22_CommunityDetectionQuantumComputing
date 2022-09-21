import numpy as np
import scipy.sparse as sps

from CommunityDetection import Community
from recsys.Data_manager.DataSplitter_Holdout import DataSplitter_Holdout
from recsys.Recommenders.Recommender_utils import reshapeSparse


def load_data(data_reader, split_quota=None, user_wise=True, make_implicit=True, threshold=None):
    print('Loading data...')

    if split_quota is None:
        split_quota = [70, 10, 20]

    data_splitter = DataSplitter_Holdout(data_reader, split_quota, user_wise=user_wise)
    data_splitter.load_data()

    urm_train, urm_validation, urm_test = data_splitter.get_holdout_split()

    if make_implicit:
        urm_train = explicit_to_implicit_urm(urm_train, threshold=threshold)
        urm_validation = explicit_to_implicit_urm(urm_validation, threshold=threshold)
        urm_test = explicit_to_implicit_urm(urm_test, threshold=threshold)

    return urm_train, urm_validation, urm_test  # , var_mapping


def merge_sparse_matrices(matrix_a, matrix_b):
    assert matrix_a.shape == matrix_b.shape, "The two matrices have different shape, they should not be merged."

    matrix_a = matrix_a.tocoo()
    matrix_b = matrix_b.tocoo()

    data_a = matrix_a.data
    row_a = matrix_a.row
    col_a = matrix_a.col

    data_b = matrix_b.data
    row_b = matrix_b.row
    col_b = matrix_b.col

    data = np.concatenate((data_a, data_b))
    row = np.concatenate((row_a, row_b))
    col = np.concatenate((col_a, col_b))

    matrix = sps.coo_matrix((data, (row, col)))

    n_users = max(matrix_a.shape[0], matrix_b.shape[0])
    n_items = max(matrix_a.shape[1], matrix_b.shape[1])
    new_shape = (n_users, n_items)

    matrix = reshapeSparse(matrix, new_shape)

    return matrix


def explicit_to_implicit_urm(urm, threshold=None):
    urm_data = urm.data

    if threshold is not None:
        urm_data_mask = urm_data >= threshold
        urm_data = urm_data_mask.astype(int)
    else:
        urm_data = np.ones_like(urm_data)

    urm.data = urm_data
    urm.eliminate_zeros()
    return urm


def get_community_urm(urm, community: Community, filter_users=True, filter_items=True, remove=False):
    new_urm = urm.copy()
    n_users, n_items = urm.shape
    new_users = np.arange(n_users)
    new_items = np.arange(n_items)

    if filter_users:
        users = community.user_mask
        new_users = new_users[users]
        if remove:
            new_urm = new_urm[users, :]
        else:
            users = np.logical_not(users)
            new_urm[users, :] = 0

    if filter_items:
        items = community.item_mask
        new_items = new_items[items]
        if remove:
            new_urm = new_urm[:, items]
        else:
            items = np.logical_not(items)
            new_urm[:, items] = 0

    new_urm.eliminate_zeros()
    return new_urm, new_users, new_items
