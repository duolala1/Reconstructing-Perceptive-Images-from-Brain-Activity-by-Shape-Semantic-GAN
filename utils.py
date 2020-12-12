import torch
import numpy as np


def samples_in_dataset(raw_labels, obj_labels_list):
    index_list = []
    index = 0
    for rlb in raw_labels:
        if rlb in obj_labels_list:
            index_list.append(index)
        index += 1
    index_list = np.asarray(index_list)
    return index_list
