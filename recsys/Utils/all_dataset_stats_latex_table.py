"""
Created on 16/06/2020

@author: Anonymous
"""

import os

def all_dataset_stats_latex_table(URM_all, dataset_name, file_path):

    if not os.path.isfile(file_path):
        data_file = open(file_path, "w")
        data_file.write("Dataset name \t &  Interactions  \t & Items  \t & Users  \t & Density \\\\\n")
        data_file.close()

    data_file = open(file_path, "a")
    data_file.write("{} \t & {}  \t & {}  \t & {}  \t & {:.1E} \\\\\n".format(
        dataset_name,
        URM_all.nnz,
        URM_all.shape[1],
        URM_all.shape[0],
        URM_all.nnz / (URM_all.shape[1] * URM_all.shape[0])
    ))