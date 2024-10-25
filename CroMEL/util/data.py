import numpy
import pandas
import itertools
import os
import math
from tqdm import tqdm
from pymatgen.core import Structure
from chemparse import parse_formula
from util.chem import get_composition_vec, get_composition_graph, get_crystal_graph


def load_dataset(path_metadata, path_structs, elem_attrs, idx_struct, idx_target, atomic_cutoff=4.0):
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


def load_dataset_by_comb(path_metadata, path_structs, elem_attrs, idx_struct, idx_target, num_elems, atomic_cutoff=4.0):
    metadata = pandas.read_excel(path_metadata).values.tolist()[:1000]
    dataset = list()
    comp_graphs = list()

    for i in tqdm(range(0, len(metadata))):
        target = metadata[i][idx_target]
        path_struct = '{}/{}.cif'.format(path_structs, metadata[i][idx_struct])

        if math.isnan(target):
            print('target', metadata[i][idx_struct])
            continue

        if not os.path.isfile(path_struct):
            print('struct', path_struct)
            continue

        struct = Structure.from_file(path_struct)
        if len(parse_formula(struct.composition.reduced_formula)) != num_elems:
            continue

        cg = get_crystal_graph(struct=struct,
                               elem_attrs=elem_attrs,
                               composition_vec=get_composition_vec(struct.composition.reduced_formula, elem_attrs),
                               y=target,
                               idx=i,
                               atomic_cutoff=atomic_cutoff)
        comp_graph = get_composition_graph(struct.composition.reduced_formula, elem_attrs)

        if cg is not None:
            dataset.append(cg)
            comp_graphs.append(comp_graph)

    return dataset, comp_graphs


def split_dataset(dataset, ratio_train=0.8, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.random.permutation(len(dataset))
    n_data_train = int(ratio_train * len(dataset))
    dataset_train = [dataset[idx] for idx in idx_rand[:n_data_train]]
    dataset_test = [dataset[idx] for idx in idx_rand[n_data_train:]]

    return dataset_train, dataset_test


def get_k_fold(dataset, n_folds, idx_fold, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.array_split(numpy.random.permutation(dataset.shape[0]), n_folds)
    idx_train = list(itertools.chain.from_iterable(idx_rand[:idx_fold] + idx_rand[idx_fold + 1:]))
    idx_test = idx_rand[idx_fold]

    return [dataset[idx] for idx in idx_train], [dataset[idx] for idx in idx_test]
