import numpy as np

from CommunityDetection.Community import Community
from utils.DataIO import DataIO
from utils.types import List, NDArray, Optional


class Communities:

    def __init__(self, user_mask: NDArray[bool], item_mask: NDArray[bool], user_index: NDArray[int] = None,
                 item_index: NDArray[int] = None):
        self.user_mask = user_mask.astype(bool)
        self.item_mask = item_mask.astype(bool)

        self.n_users = len(user_mask)
        self.n_items = len(item_mask)

        if user_index is None:
            user_index = np.arange(self.n_users)
        if item_index is None:
            item_index = np.arange(self.n_items)

        self.user_index = user_index
        self.item_index = item_index

        c0_users = user_index[~self.user_mask]
        c0_items = item_index[~self.item_mask]
        self.c0: Community = Community(c0_users, c0_items, ~self.user_mask, ~self.item_mask)

        c1_users = user_index[self.user_mask]
        c1_items = item_index[self.item_mask]
        self.c1: Community = Community(c1_users, c1_items, self.user_mask, self.item_mask)

        self.s0: Optional[Communities] = None
        self.s1: Optional[Communities] = None
        self.num_iters = 0

    def iter(self, n_iter: int = None):
        if n_iter is None:
            n_iter = self.num_iters
        if n_iter == 0 or (self.s0 is None and self.s1 is None):
            yield self.c0
            yield self.c1
        else:
            yield from self.s0.iter(n_iter - 1)
            yield from self.s1.iter(n_iter - 1)

    def add_iteration(self, communities: List):
        self.num_iters += 1
        n_communities = len(communities)
        assert n_communities == 2 ** self.num_iters, \
            'Cannot add a number of communities different from a power of 2.'

        if self.s0 is None and self.s1 is None:
            self.s0 = communities[0]
            self.s1 = communities[1]

            self.s0.__adjust_masks(self.n_users, self.n_items)
            self.s1.__adjust_masks(self.n_users, self.n_items)
        else:
            half_communities = n_communities // 2
            self.s0.add_iteration(communities[:half_communities])
            self.s1.add_iteration(communities[half_communities:])

    def __adjust_masks(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items
        self.c0.adjust_masks(n_users, n_items)
        self.c1.adjust_masks(n_users, n_items)

    @classmethod
    def load(cls, folder_path, file_name='communities', n_iter=0, n_comm=None, folder_suffix=''):
        comm_folder_path = get_community_folder_path(folder_path, n_iter=n_iter, folder_suffix=folder_suffix)
        dataIO = DataIO(comm_folder_path)

        comm_file_suffix = f'{n_comm:02d}' if n_comm is not None else ''
        comm_file_name = f'{file_name}{comm_file_suffix}'
        data_dict = dataIO.load_data(comm_file_name)

        communities = Communities(data_dict['user_mask'], data_dict['item_mask'],
                                  data_dict['user_index'], data_dict['item_index'])
        communities.num_iters = data_dict['num_iters']
        communities.__adjust_masks(data_dict['n_users'], data_dict['n_items'])

        if communities.num_iters > 0:
            assert n_iter is not None, 'Cannot load more iterations without the correct parameters.'

            if n_comm is None:
                n_comm = 0

            try:
                communities.s0 = Communities.load(folder_path, file_name, n_iter + 1, 2 * n_comm, folder_suffix)
            except FileNotFoundError:
                communities.s0 = None
            try:
                communities.s1 = Communities.load(folder_path, file_name, n_iter + 1, 2 * n_comm + 1, folder_suffix)
            except FileNotFoundError:
                communities.s1 = None

            s0_num_iters = 0 if communities.s0 is None else communities.s0.num_iters + 1
            s1_num_iters = 0 if communities.s1 is None else communities.s1.num_iters + 1
            communities.num_iters = min(s0_num_iters, s1_num_iters)

        return communities

    def save(self, folder_path, file_name='communities', n_iter=0, n_comm=None, folder_suffix=''):
        comm_folder_path = get_community_folder_path(folder_path, n_iter=n_iter, folder_suffix=folder_suffix)
        dataIO = DataIO(comm_folder_path)

        comm_file_suffix = f'{n_comm:02d}' if n_comm is not None else ''
        comm_file_name = f'{file_name}{comm_file_suffix}'

        data_dict_to_save = {
            'user_mask': self.user_mask,
            'item_mask': self.item_mask,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'user_index': self.user_index,
            'item_index': self.item_index,
            'num_iters': self.num_iters,
        }

        dataIO.save_data(comm_file_name, data_dict_to_save)

        if self.num_iters > 0:
            if n_comm is None:
                n_comm = 0
            self.s0.save(folder_path, file_name, n_iter + 1, 2 * n_comm, folder_suffix)
            self.s1.save(folder_path, file_name, n_iter + 1, 2 * n_comm + 1, folder_suffix)

    def reset_from_iter(self, n_iter):
        assert n_iter != 0, 'Should not reset from iteration 0.'
        assert n_iter > 0, 'Cannot use a negative iteration value.'

        if n_iter == 1:
            self.s0 = None
            self.s1 = None
            self.num_iters = 0
        else:
            if self.s0 is None or self.s1 is None:
                print('Requested a number of iterations greater than the ones computed.')
                self.s0 = None
                self.s1 = None
                self.num_iters = 0
            else:
                self.s0.reset_from_iter(n_iter - 1)
                self.s1.reset_from_iter(n_iter - 1)
                self.num_iters = min(self.s0.num_iters, self.s1.num_iters) + 1

    def load_from_iter(self, starting_iter, folder_path, file_name='communities', folder_suffix='', n_iter=0, n_comm=0):
        assert starting_iter != 0, 'Should not partially load from iteration 0. Use Communities.load() in such case.'
        assert starting_iter > 0, 'Cannot use a negative iteration value.'

        if starting_iter == (n_iter + 1):
            self.s0 = Communities.load(folder_path, file_name=file_name, n_iter=n_iter, n_comm=n_comm,
                                       folder_suffix=folder_suffix)
            self.s1 = Communities.load(folder_path, file_name=file_name, n_iter=n_iter, n_comm=n_comm + 1,
                                       folder_suffix=folder_suffix)
            self.num_iters = min(self.s0.num_iters, self.s1.num_iters) + 1
        else:
            self.s0.load_from_iter(starting_iter, folder_path, file_name=file_name, folder_suffix=folder_suffix,
                                   n_iter=n_iter + 1, n_comm=2 * n_comm)
            self.s1.load_from_iter(starting_iter, folder_path, file_name=file_name, folder_suffix=folder_suffix,
                                   n_iter=n_iter + 1, n_comm=2 * n_comm + 1)
            self.num_iters = min(self.s0.num_iters, self.s1.num_iters) + 1

    def save_from_iter(self, starting_iter, folder_path, file_name='communities', folder_suffix='', n_iter=0, n_comm=0):
        assert starting_iter != 0, 'Should not partially save from iteration 0. Use Communities.save() in such case.'
        assert starting_iter > 0, 'Cannot use a negative iteration value.'

        if starting_iter == (n_iter + 1):
            self.s0.save(folder_path, file_name=file_name, n_iter=n_iter + 1, n_comm=n_comm,
                         folder_suffix=folder_suffix)
            self.s1.save(folder_path, file_name=file_name, n_iter=n_iter + 1, n_comm=n_comm + 1,
                         folder_suffix=folder_suffix)
        else:
            self.s0.save_from_iter(starting_iter, folder_path, file_name=file_name, folder_suffix=folder_suffix,
                                   n_iter=n_iter + 1, n_comm=2 * n_comm)
            self.s1.save_from_iter(starting_iter, folder_path, file_name=file_name, folder_suffix=folder_suffix,
                                   n_iter=n_iter + 1, n_comm=2 * (n_comm + 1))


def get_community_folder_path(folder_path, n_iter=None, n_comm=None, folder_suffix=''):
    return f'{folder_path}' \
           f'{f"iter{n_iter:02d}/" if n_iter is not None else ""}' \
           f'{folder_suffix}' \
           f'{f"c{n_comm:02d}/" if n_comm is not None else ""}'
