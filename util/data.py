import torch
import numpy
import pandas
import os
import math
import warnings
from itertools import chain
from tqdm import tqdm
from pymatgen.core import Structure
from util.chem import get_composition_vec, get_crystal_graph


warnings.filterwarnings(action='ignore')


class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ExpDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.x = torch.tensor(numpy.vstack([d.x for d in self.data]), dtype=torch.float)
        self.y = torch.tensor(numpy.vstack([d.y for d in self.data]), dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_k_folds(self, num_folds, random_seed=None):
        if random_seed is not None:
            numpy.random.seed(random_seed)

        idx_rand = numpy.array_split(numpy.random.permutation(len(self.data)), num_folds)
        sub_datasets = list()
        for i in range(0, num_folds):
            sub_datasets.append([self.data[idx] for idx in idx_rand[i]])

        k_folds = list()
        for i in range(0, num_folds):
            dataset_train = ExpDataset(list(chain.from_iterable(sub_datasets[:i] + sub_datasets[i + 1:])))
            dataset_test = ExpDataset(sub_datasets[i])
            k_folds.append([dataset_train, dataset_test])

        return k_folds


def load_calc_dataset(path_metadata, path_structs, elem_attrs, idx_struct, idx_target, atomic_cutoff=4.0):
    metadata = pandas.read_excel(path_metadata).values.tolist()
    dataset = list()

    for i in tqdm(range(0, len(metadata))):
        try:
            target = metadata[i][idx_target]
            path_struct = '{}/{}.cif'.format(path_structs, metadata[i][idx_struct])

            if math.isnan(target):
                print('target', metadata[i][idx_struct])
                continue

            if not os.path.isfile(path_struct):
                print('struct', path_struct)
                continue

            struct = Structure.from_file(path_struct)
            cg = get_crystal_graph(struct=struct,
                                   elem_attrs=elem_attrs,
                                   composition_vec=get_composition_vec(struct.composition.reduced_formula, elem_attrs),
                                   y=target,
                                   idx=i,
                                   atomic_cutoff=atomic_cutoff)

            if cg is not None:
                dataset.append(cg)
        except ValueError:
            print('Invalid structure: {}'.format(metadata[i][idx_struct]))

    return dataset


def load_exp_dataset(path_dataset, elem_attrs, idx_composition, idx_target, log_target):
    data_list = pandas.read_excel(path_dataset).values.tolist()
    data = list()

    for i in tqdm(range(0, len(data_list))):
        if pandas.isnull(data_list[i][idx_target]):
            continue

        comp_vec = get_composition_vec(data_list[i][idx_composition], elem_attrs)
        if log_target:
            data.append(Data(comp_vec, numpy.log10(data_list[i][idx_target])))
        else:
            data.append(Data(comp_vec, data_list[i][idx_target]))

    return ExpDataset(data)
