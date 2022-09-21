import numpy.typing as npt

from utils.DataIO import DataIO


class BaseCommunityDetection:
    filter_users = True
    filter_items = True
    is_qubo = False
    name = 'BaseCommunityDetection'

    def __init__(self, urm: npt.ArrayLike):
        self.urm = urm

        self._fit_time = None

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            f'The {self.fit.__name__} method is not implemented '
            f'for the abstract base class {self.__class__.__name__}.')

    def run(self, *args, **kwargs):
        raise NotImplementedError(
            f'The {self.run.__name__} method is not implemented '
            f'for the abstract base class {self.__class__.__name__}.')

    def save_model(self, folder_path, file_name):
        raise NotImplementedError(
            f'The {self.save_model.__name__} method is not implemented '
            f'for the abstract base class {self.__class__.__name__}.')

    def load_model(self, folder_path, file_name):
        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
            self.__setattr__(attrib_name, data_dict[attrib_name])
