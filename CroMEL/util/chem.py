import numpy
import json
import torch
from chemparse import parse_formula
from torch_geometric.data import Data
from sklearn.metrics import pairwise_distances


atom_nums = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
}
atom_syms = {v: k for k, v in atom_nums.items()}


def load_elem_attrs(path_elem_attr):
    with open(path_elem_attr) as json_file:
        elem_attr = json.load(json_file)

    return numpy.vstack([elem_attr[elem] for elem in atom_nums.keys()])


def get_composition_vec(composition, elem_attrs):
    elem_dict = parse_formula(composition)
    wt_sum_feats = numpy.zeros(elem_attrs.shape[1])
    list_atom_feats = list()
    sum_elem_nums = sum([float(elem_dict[e]) for e in elem_dict.keys()])

    for e in elem_dict.keys():
        atom_feats = elem_attrs[atom_nums[e] - 1, :]
        list_atom_feats.append(atom_feats)
        wt_sum_feats += (float(elem_dict[e]) / sum_elem_nums) * atom_feats
    form_atom_feats = numpy.vstack(list_atom_feats)
    composition_vec = numpy.hstack([wt_sum_feats,
                                    numpy.min(form_atom_feats, axis=0),
                                    numpy.max(form_atom_feats, axis=0)])

    return composition_vec


def get_composition_graph(composition, elem_attrs):
    elem_dict = parse_formula(composition)
    node_feats = list()
    edges = list()

    for e in elem_dict.keys():
        node_feats.append(elem_attrs[atom_nums[e] - 1, :])

    for i in range(0, len(node_feats)):
        for j in range(0, len(node_feats)):
            edges.append([i, j])

    return Data(x=torch.tensor(numpy.vstack(node_feats), dtype=torch.float),
                edge_index=torch.tensor(numpy.vstack(edges), dtype=torch.long).t().contiguous())


def get_crystal_graph(struct, elem_attrs, composition_vec, y, idx=-1, atomic_cutoff=4.0):
    try:
        rbf_means = numpy.linspace(start=1.0, stop=atomic_cutoff, num=64)
        atom_coord, atom_feats, ans = get_atom_info(struct, elem_attrs, atomic_cutoff)
        bonds, bond_feats = get_bond_info(atom_coord, rbf_means, atomic_cutoff)

        if bonds is None:
            return None

        return Data(x=torch.tensor(atom_feats, dtype=torch.float),
                    edge_index=torch.tensor(bonds, dtype=torch.long).t().contiguous(),
                    edge_attr=torch.tensor(bond_feats, dtype=torch.float),
                    composition_vec=torch.tensor(composition_vec, dtype=torch.float).view(1, -1),
                    y=torch.tensor(y, dtype=torch.float).view(1, 1),
                    idx=torch.tensor(idx, dtype=torch.long).view(1, 1))
    except AssertionError:
        return None


def get_atom_info(crystal, elem_attrs, atomic_cutoff):
    atoms = list(crystal.atomic_numbers)
    atom_coord = list()
    atom_feats = list()
    list_nbrs = crystal.get_all_neighbors(atomic_cutoff)

    coords = dict()
    for coord in list(crystal.cart_coords):
        coord_key = ','.join(list(coord.astype(str)))
        coords[coord_key] = True

    for i in range(0, len(list_nbrs)):
        nbrs = list_nbrs[i]

        for j in range(0, len(nbrs)):
            coord_key = ','.join(list(nbrs[j][0].coords.astype(str)))
            if coord_key not in coords.keys():
                coords[coord_key] = True
                atoms.append(atom_nums[nbrs[j][0].species_string])

    for coord in coords.keys():
        atom_coord.append(numpy.array([float(x) for x in coord.split(',')]))
    atom_coord = numpy.vstack(atom_coord)

    for i in range(0, len(atoms)):
        atom_feats.append(elem_attrs[atoms[i]-1, :])
    atom_feats = numpy.vstack(atom_feats).astype(float)

    return atom_coord, atom_feats, atoms


def get_bond_info(atom_coord, rbf_means, atomic_cutoff):
    bonds = list()
    bond_feats = list()
    pdist = pairwise_distances(atom_coord)

    for i in range(0, atom_coord.shape[0]):
        for j in range(0, atom_coord.shape[0]):
            if i != j and pdist[i, j] < atomic_cutoff:
                bonds.append([i, j])
                bond_feats.append(rbf(numpy.full(rbf_means.shape[0], pdist[i, j]), rbf_means, beta=0.5))

    if len(bonds) == 0:
        return None, None
    else:
        bonds = numpy.vstack(bonds)
        bond_feats = numpy.vstack(bond_feats)

        return bonds, bond_feats


def rbf(data, mu, beta):
    return numpy.exp(-(data - mu)**2 / beta**2)
