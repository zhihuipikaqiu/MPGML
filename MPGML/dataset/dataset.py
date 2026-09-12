"""Molecular graphs, assay splits, and cached auxiliary features for MPGML."""

import csv
import os.path as osp
import pickle

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from .mol_features import allowable_features
from .molecular_features import (
    calculate_fingerprint,
    calculate_molecular_descriptors,
    get_descriptor_dim,
    get_fingerprint_dim,
    smiles_to_mol,
)


class FewshotMolDataset(InMemoryDataset):
    """Load a local assay CSV and split tasks by their label-column order.

    Raw files belong in ``root/name/name.csv``. Missing labels are stored as
    NaN and excluded from the per-task sampling pools. Graphs and molecular
    feature matrices are cached under ``root/name/processed``.
    """

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx, train_tasks, test_tasks]
    names = {
        'pcba': ['PCBA', 'pcba', 'pcba', -1, slice(0, 128), 118, 10],
        'muv': ['MUV', 'muv', 'muv', -1, slice(0, 17), 12, 5],
        'tox21': ['Tox21', 'tox21', 'tox21', -1, slice(0, 12), 9, 3],
        'sider': ['SIDER', 'sider', 'sider', 0, slice(1, 28), 21, 6],

        # ToxCast assay-provider groups.
        'toxcast-APR': ['ToxCast-APR', 'toxcast-APR', 'toxcast-APR', 0, slice(1, 44), 33, 10],
        'toxcast-ATG': ['ToxCast-ATG', 'toxcast-ATG', 'toxcast-ATG', 0, slice(1, 147), 106, 40],
        'toxcast-BSK': ['ToxCast-BSK', 'toxcast-BSK', 'toxcast-BSK', 0, slice(1, 116), 84, 31],
        'toxcast-CEETOX': ['ToxCast-CEETOX', 'toxcast-CEETOX', 'toxcast-CEETOX', 0, slice(1, 15), 10, 4],
        'toxcast-CLD': ['ToxCast-CLD', 'toxcast-CLD', 'toxcast-CLD', 0, slice(1, 20), 14, 5],
        'toxcast-NVS': ['ToxCast-NVS', 'toxcast-NVS', 'toxcast-NVS', 0, slice(1, 140), 100, 39],
        'toxcast-OT': ['ToxCast-OT', 'toxcast-OT', 'toxcast-OT', 0, slice(1, 16), 11, 4],
        'toxcast-TOX21': ['ToxCast-TOX21', 'toxcast-TOX21', 'toxcast-TOX21', 0, slice(1, 101), 80, 20],
        'toxcast-Tanguay': ['ToxCast-Tanguay', 'toxcast-Tanguay', 'toxcast-Tanguay', 0, slice(1, 19), 13, 5],
    }

    @classmethod
    def valid_names(cls):
        return tuple(cls.names.keys())

    @classmethod
    def canonical_name(cls, name):
        name_map = {dataset_name.lower(): dataset_name for dataset_name in cls.valid_names()}
        return name_map.get(str(name).lower())

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None, use_fingerprint=False, use_descriptor=False,
                 fp_type='mixed', rebuild_feature_cache=False):

        self.name = self.canonical_name(name)
        if self.name is None:
            valid_names = ', '.join(self.valid_names())
            raise ValueError(f'Unknown dataset "{name}". Valid datasets: {valid_names}.')
        self.use_fingerprint = use_fingerprint
        self.use_descriptor = use_descriptor
        self.fp_type = fp_type.lower()
        self.rebuild_feature_cache = rebuild_feature_cache
        self.fingerprint_matrix = None
        self.descriptor_matrix = None
        self.descriptor_mean = None
        self.descriptor_std = None
        super(FewshotMolDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.n_task_train, self.n_task_test = self.names[self.name][5], self.names[self.name][6]
        self.total_tasks = self.n_task_train + self.n_task_test
        if self.name != 'pcba':
            self.train_task_range = list(range(self.n_task_train))
            self.test_task_range = list(range(self.n_task_train, self.n_task_train + self.n_task_test))
        else:
            self.train_task_range = list(range(5, self.total_tasks - 5))
            self.test_task_range = list(range(5)) + list(range(self.total_tasks - 5, self.total_tasks))

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        with open(self.processed_paths[1], 'rb') as handle:
            self.index_list = pickle.load(handle)
        with open(self.processed_paths[2], 'rb') as handle:
            self.y_matrix = np.load(handle)
        if self.use_fingerprint or self.use_descriptor:
            self._load_or_build_feature_cache()

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt', 'index_list.pt', 'label_matrix.npz'

    def _raw_rows(self):
        """Stream parsed rows without retaining a second copy of the raw CSV."""
        with open(self.raw_paths[0], newline='', encoding='utf-8-sig') as handle:
            reader = csv.reader(handle)
            next(reader, None)  # Assay names in the header define the task order.
            for row in reader:
                if row:
                    yield row

    def process(self):
        data_list = []
        y_list = []
        for line in tqdm(self._raw_rows(), desc=f'graphs-{self.name}'):
            smiles = line[self.names[self.name][3]]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            Chem.Kekulize(mol)

            ys = line[self.names[self.name][4]]
            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            xs = []
            for atom in mol.GetAtoms():
                x = []
                x.append(allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum()))
                x.append(allowable_features['possible_chirality_list'].index(atom.GetChiralTag()))
                xs.append(x)

            x = torch.tensor(xs, dtype=torch.long).view(-1, 2)

            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                e = []
                e.append(allowable_features['possible_bonds'].index(bond.GetBondType()))
                e.append(allowable_features['possible_bond_dirs'].index(bond.GetBondDir()))

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 2)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        smiles=smiles, id=len(data_list))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            y_list.append(data.y.view(-1).tolist())

        if not data_list:
            raise ValueError(f'No molecules remain after processing {self.raw_paths[0]}.')

        y_matrix = np.array(y_list)
        index_list = []  # index_list[task_id][class_label] -> molecule indices.
        for task_i in range(y_matrix.shape[1]):
            task_i_label_values = y_matrix[:, task_i]
            class1_index = np.nonzero(task_i_label_values > 0.5)[0].tolist()
            class0_index = np.nonzero(task_i_label_values < 0.5)[0].tolist()
            index_list.append([class0_index, class1_index])

        torch.save(self.collate(data_list), self.processed_paths[0])
        with open(self.processed_paths[1], 'wb') as handle:
            pickle.dump(index_list, handle)
        # Preserve the historical filename; the payload is an NPY array.
        with open(self.processed_paths[2], 'wb') as handle:
            np.save(handle, y_matrix)

    def get(self, idx):
        data = super().get(idx)
        mol_id = self._get_mol_id(data, idx)

        if self.use_fingerprint:
            if self.fingerprint_matrix is None:
                raise RuntimeError('Fingerprint cache is not loaded.')
            data.fp = self.fingerprint_matrix[mol_id].view(1, -1)

        if self.use_descriptor:
            if self.descriptor_matrix is None:
                raise RuntimeError('Descriptor cache is not loaded.')
            data.desc = self.descriptor_matrix[mol_id].view(1, -1)

        return data

    def _get_mol_id(self, data, idx):
        if hasattr(data, 'id'):
            mol_id = data.id
            if torch.is_tensor(mol_id):
                return int(mol_id.view(-1)[0].item())
            return int(mol_id)
        return int(idx)

    @property
    def feature_cache_path(self):
        return osp.join(self.processed_dir, f'molecular_features_{self.fp_type}.pt')

    def _load_or_build_feature_cache(self):
        payload = None
        if (not self.rebuild_feature_cache) and osp.exists(self.feature_cache_path):
            try:
                payload = torch.load(self.feature_cache_path, map_location='cpu', weights_only=False)
                if not self._is_feature_cache_valid(payload):
                    payload = None
            except Exception:
                payload = None

        if payload is None:
            payload = self._build_feature_cache()
            torch.save(payload, self.feature_cache_path)

        if self.use_fingerprint:
            self.fingerprint_matrix = payload['fingerprints'].float()
        if self.use_descriptor:
            self.descriptor_matrix = payload['descriptors'].float()
            self.descriptor_mean = payload.get('descriptor_mean')
            self.descriptor_std = payload.get('descriptor_std')

    def _is_feature_cache_valid(self, payload):
        if not isinstance(payload, dict):
            return False
        if payload.get('dataset') != self.name or payload.get('fp_type') != self.fp_type:
            return False
        if payload.get('num_molecules') != len(self):
            return False
        if self.use_fingerprint:
            fps = payload.get('fingerprints')
            if fps is None or fps.shape != (len(self), get_fingerprint_dim(self.fp_type)):
                return False
        if self.use_descriptor:
            descs = payload.get('descriptors')
            if descs is None or descs.shape != (len(self), get_descriptor_dim()):
                return False
        return True

    def _build_feature_cache(self):
        """Compute auxiliary inputs; descriptor statistics use all molecules."""
        fingerprints = []
        descriptors = []

        for idx in tqdm(range(len(self)), desc=f'features-{self.name}-{self.fp_type}'):
            data = super().get(idx)
            smiles = data.smiles
            mol = smiles_to_mol(smiles)
            if self.use_fingerprint:
                fingerprints.append(calculate_fingerprint(mol, self.fp_type))
            if self.use_descriptor:
                descriptors.append(calculate_molecular_descriptors(mol))

        payload = {
            'dataset': self.name,
            'fp_type': self.fp_type,
            'num_molecules': len(self),
        }

        if self.use_fingerprint:
            payload['fingerprints'] = torch.from_numpy(
                np.asarray(fingerprints, dtype=np.float32)
            )

        if self.use_descriptor:
            desc_arr = np.asarray(descriptors, dtype=np.float32)
            desc_arr = np.nan_to_num(desc_arr, nan=0.0, posinf=0.0, neginf=0.0)
            # Dataset-wide, label-free standardization is part of this version's
            # preprocessing protocol; it is not fitted on support episodes.
            desc_mean = desc_arr.mean(axis=0, keepdims=True)
            desc_std = desc_arr.std(axis=0, keepdims=True)
            desc_std[desc_std < 1e-6] = 1.0
            desc_arr = (desc_arr - desc_mean) / desc_std
            payload['descriptors'] = torch.from_numpy(desc_arr.astype(np.float32))
            payload['descriptor_mean'] = torch.from_numpy(desc_mean.astype(np.float32))
            payload['descriptor_std'] = torch.from_numpy(desc_std.astype(np.float32))

        return payload

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))
