import numpy
import pandas
import torch
import itertools
from tqdm import tqdm
from util.chem import get_composition_vec


def load_dataset(path_dataset, elem_attrs, idx_composition, idx_target, log_target):
    metadata = pandas.read_excel(path_dataset).values.tolist()
    dataset = list()

    for i in tqdm(range(0, len(metadata))):
        composition_vec = get_composition_vec(metadata[i][idx_composition], elem_attrs)

        if pandas.isnull(metadata[i][idx_target]):
            continue

        if log_target:
            dataset.append(numpy.hstack([composition_vec, numpy.log(metadata[i][idx_target])]))
        else:
            dataset.append(numpy.hstack([composition_vec, metadata[i][idx_target]]))

    return torch.tensor(numpy.vstack(dataset), dtype=torch.float)


def get_k_fold(dataset, n_folds, idx_fold, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.array_split(numpy.random.permutation(dataset.shape[0]), n_folds)
    idx_train = list(itertools.chain.from_iterable(idx_rand[:idx_fold] + idx_rand[idx_fold + 1:]))
    idx_test = idx_rand[idx_fold]

    return dataset[idx_train], dataset[idx_test]


def split_dataset(dataset, ratio_train=0.8, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.random.permutation(len(dataset))
    idx_train = idx_rand[:int(ratio_train * len(dataset))]
    idx_test = idx_rand[int(0.8 * len(dataset)):]

    return dataset[idx_train], dataset[idx_test]
