import numpy as np
from utils.types import NDArray


class Community:

    def __init__(self, users, items, user_mask, item_mask):
        self.users: NDArray[int] = users
        self.items: NDArray[int] = items

        self.user_mask: NDArray[bool] = user_mask
        self.item_mask: NDArray[bool] = item_mask

    def __str__(self):
        return f'users: {self.users}\n' \
               f'items: {self.items}\n' \
               f'user_mask: {self.user_mask}\n' \
               f'item_mask: {self.item_mask}'

    def adjust_masks(self, n_users, n_items):
        self.user_mask = np.zeros(n_users, dtype=bool)
        self.user_mask[self.users] = True

        self.item_mask = np.zeros(n_items, dtype=bool)
        self.item_mask[self.items] = True
