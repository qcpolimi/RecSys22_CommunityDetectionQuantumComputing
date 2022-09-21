#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04/07/19

@author: Anonymous
"""

import multiprocessing, traceback
import numpy as np

def _load_data_and_get_size(dataset_class):

    try:

        dataset_object = dataset_class()
        dataset_object.load_data()

        URM_all = dataset_object.get_URM_all()

        size = {"interactions": URM_all.nnz,
                "users": URM_all.shape[0],
                "items": URM_all.shape[1]}

    except:
        traceback.print_exc()
        size = None


    return {dataset_class: size}



def sort_dataset_on_size(dataset_class_list, sort_on = "interactions"):

    assert sort_on in ["interactions", "users", "items"]

    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    dataset_size_list = pool.map(_load_data_and_get_size, dataset_class_list)

    dataset_size_dict = {}
    for dataset_dict in dataset_size_list:
        dataset_size_dict = {**dataset_size_dict, **dataset_dict}

    pool.close()
    pool.join()

    size_list = []

    for dataset_class in dataset_class_list:
        if dataset_size_dict[dataset_class] is not None:
            size_list.append(dataset_size_dict[dataset_class][sort_on])
        else:
            size_list.append(np.inf)

    size_argsort = np.argsort(np.array(size_list))

    dataset_class_list = [dataset_class_list[index] for index in size_argsort]

    return dataset_class_list
