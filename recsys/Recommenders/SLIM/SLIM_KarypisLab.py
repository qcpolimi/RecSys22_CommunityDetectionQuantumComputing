#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anonymous
"""


from recsys.Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from recsys.Recommenders.BaseTempFolder import BaseTempFolder
import scipy.sparse as sps
import time, subprocess, os
from recsys.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from recsys.Recommenders.Recommender_utils import similarityMatrixTopK



class SLIM_KarypisLab(BaseItemSimilarityMatrixRecommender, BaseTempFolder):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.

    See:
        https://www-users.cs.umn.edu/~ningx005/slim/html/index.html
    """

    RECOMMENDER_NAME = "SLIM_KarypisLab_Recommender"

    def __init__(self, URM_train, recompile = False):

        super(SLIM_KarypisLab, self).__init__(URM_train)
        self._this_file_path = os.path.dirname(__file__)

        if recompile:
            self._compile()



    def fit(self, l1_penalty=0.1, l2_penalty=0.1,
            topK = 100,
            optimization_threshold = 1e-5,
            iterations_threshold = 1e5,
            temp_file_folder = None,
            train_file_name = "URM_train_file.csv",
            model_file_name = "SLIM_KarypisLab_W_sparse.csv"):

        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)
        self.train_file_name = train_file_name
        self.model_file_name = model_file_name

        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.optimization_threshold = optimization_threshold
        self.iterations_threshold = iterations_threshold


        self._build_SLIM_URM_file()

        self._run_training()

        self._read_W_sparse()

        self.W_sparse = similarityMatrixTopK(self.W_sparse, k=topK)

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)



    def _compile(self):

        decompress_command = ["tar", "-xzf", "slim-1.0.tar.gz"]
        output = subprocess.check_output(' '.join(decompress_command), shell=True, cwd=self._this_file_path + "/SLIM_KarypisLab")

        command_list = [
            # ["tar", "-xzf", "slim-1.0.tar.gz"],
            # ["cd", "slim-1.0"],
            # ["cd", "build"],
            ["cmake", ".."],
            ["make"],
            ["make", "install"]
            ]

        for command in command_list:
            output = subprocess.check_output(' '.join(command), shell=True, cwd=self._this_file_path + "/SLIM_KarypisLab/slim-1.0/build")





    def _build_SLIM_URM_file(self):

        URM_train_file = open(self.temp_file_folder + self.train_file_name, "w")

        n_users = self.URM_train.shape[0]

        self._print("Building sparse URM file")

        start_time = time.time()
        start_time_print = time.time()

        for currentUser in range(n_users):

            start_pos = self.URM_train.indptr[currentUser]
            end_pos = self.URM_train.indptr[currentUser+1]

            currentUser_indices = self.URM_train.indices[start_pos:end_pos]
            currentUser_data = self.URM_train.data[start_pos:end_pos]

            for data_index in range(len(currentUser_indices)):

                # Columns are 1 indexed, so they start from one. In CSR scipy they start from zero
                URM_train_file.write("{} {}".format(currentUser_indices[data_index] +1, int(currentUser_data[data_index])))

                if data_index < len(currentUser_indices)-1:
                    URM_train_file.write(" ")

            if currentUser < n_users-1:
                URM_train_file.write("\n")


            if time.time() - start_time_print > 30:
                self._print("Processed {} ({:4.1f}%) users".format(currentUser, currentUser/n_users*100))
                start_time_print = time.time()


        URM_train_file.close()

        self._print("Building sparse URM file... done!")




    def _read_W_sparse(self):

        self._print("Reading W sparse...")

        W_sparse_file = open(self.temp_file_folder + self.model_file_name, "r")
        W_sparse_incremental = IncrementalSparseMatrix(n_cols=self.n_items, n_rows=self.n_items)

        current_row_index = 0

        for line in W_sparse_file:

            if line == "\n":
                current_row_index += 1
                continue

            line_list = line.split(" ")

            if line_list[0] == "":
                del line_list[0]

            line_list[-1] = line_list[-1].replace("\n", "")

            #line_list_col_values = [line_list[i:i + 2] for i in range(0, len(line_list), 2)]

            try:
                column_index_list = [int(line_list[i]) for i in range(0, len(line_list), 2)]
                data_list = [float(line_list[i+1]) for i in range(0, len(line_list), 2)]

            except ValueError:
                pass

            row_index_list = [current_row_index]*len(column_index_list)

            W_sparse_incremental.add_data_lists(row_index_list, column_index_list, data_list)

            current_row_index += 1



        self.W_sparse = W_sparse_incremental.get_SparseMatrix()
        self.W_sparse = sps.csr_matrix(self.W_sparse.T)

        self._print("Reading W sparse... done!")



    def _run_training(self):

        self._print("Begin training...")

        command = [self._this_file_path + "/SLIM_KarypisLab/slim-1.0/build/examples/slim_learn",
                   "-train_file={}".format(self.train_file_name),
                   "-model_file={}".format(self.model_file_name),
                   "-lambda={}".format(self.l1_penalty),
                   "-beta={}".format(self.l2_penalty),
                   "-optTol={}".format(self.optimization_threshold),
                   "-max_bcls_niters={:.0f}".format(self.iterations_threshold)
                   ]

        output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + "/" + self.temp_file_folder)

        self._print("Begin training... done!")


