# -*- coding: utf-8 -*-
"""
@Author  : Weimin Zhu
@Time    : 2021-09-28
@File    : dataset.py
"""

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import Random
from collections import defaultdict

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import DataLoader
# from torch_geometric.loader import DataLoader

from rdkit import Chem
from rdkit.Chem.BRICS import FindBRICSBonds
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

from utils import get_task_names, FRAMEWORK_BOND_SP2, FRAMEWORK_BOND_SP3
from smiles_bert import SMILESTransformer

from data_enhancement import data_enhancement

RDLogger.DisableLog('rdApp.*')


# -------------------------------------
# attentive_fp fashion featurization
# -------------------------------------
def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def _build_scaffold_sort_key(_bit_width):
    # Encode molecular framework bond-type information for deterministic group ordering
    # SP2 framework bond marker: represents aromatic / conjugated bond in molecular core
    # SP3 framework bond marker: represents saturated single-bond in molecular framework
    _mol_sp2 = chr(FRAMEWORK_BOND_SP2)
    _mol_sp3 = chr(FRAMEWORK_BOND_SP3)
    def _bit_to_marker(_val, _bits):
        _bin = format(_val, f'0{_bits}b')
        _out = ''
        for _c in _bin:
            _out += _mol_sp2 if _c == '0' else _mol_sp3
        return _out
    return _bit_to_marker(_val, _bit_width) if False else _bit_to_marker

def atom_attr(mol, explicit_H=False, use_chirality=True, pharmaco=True, scaffold=True):
    if pharmaco:
        mol = tag_pharmacophore(mol)
    if scaffold:
        mol = tag_scaffold(mol)

    feat = []
    for i, atom in enumerate(mol.GetAtoms()):
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other'
             ]) + onehot_encoding_unk(atom.GetDegree(),
                                      [0, 1, 2, 3, 4, 5, 'other']) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  onehot_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'
                  ]) + [atom.GetIsAromatic()]
        if not explicit_H:
            results = results + onehot_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + onehot_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            # print(one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')])
            except:
                results = results + [0, 0] + [atom.HasProp('_ChiralityPossible')]
        if pharmaco:
            results = results + [int(atom.GetProp('Hbond_donor'))] + [int(atom.GetProp('Hbond_acceptor'))] + \
                      [int(atom.GetProp('Basic'))] + [int(atom.GetProp('Acid'))] + \
                      [int(atom.GetProp('Halogen'))]
        if scaffold:
            results = results + [int(atom.GetProp('Scaffold'))]
        feat.append(results)

    return np.array(feat)


def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)


def bond_break(mol):
    results = np.array(sorted(list(FindBRICSBonds(mol))), dtype=np.long)

    if results.size == 0:
        cluster_idx = []
        Chem.rdmolops.GetMolFrags(mol, asMols=True, frags=cluster_idx)
        fra_edge_index, fra_edge_attr = bond_attr(mol)

    else:
        bond_to_break = results[:, 0, :]
        bond_to_break = bond_to_break.tolist()
        with Chem.RWMol(mol) as rwmol:
            for i in bond_to_break:
                rwmol.RemoveBond(*i)
        rwmol = rwmol.GetMol()
        cluster_idx = []
        Chem.rdmolops.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False, frags=cluster_idx)
        fra_edge_index, fra_edge_attr = bond_attr(rwmol)
        cluster_idx = torch.LongTensor(cluster_idx)

    return fra_edge_index, fra_edge_attr, cluster_idx


# ---------------------------------------------
# Scaffold and pharmacophore information utils
# ---------------------------------------------
# tag pharmoco features to each atom
fun_smarts = {
        'Hbond_donor': '[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]',
        'Hbond_acceptor': '[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&X2&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]',
        'Basic': '[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),$([n;X2;+0;-0])]',
        'Acid': '[C,S](=[O,S,P])-[O;H1,-1]',
        'Halogen': '[F,Cl,Br,I]'
        }
FunQuery = dict([(pharmaco, Chem.MolFromSmarts(s)) for (pharmaco, s) in fun_smarts.items()])


def tag_pharmacophore(mol):
    for fungrp, qmol in FunQuery.items():
        matches = mol.GetSubstructMatches(qmol)
        match_idxes = []
        for mat in matches:
            match_idxes.extend(mat)
        for i, atom in enumerate(mol.GetAtoms()):
            tag = '1' if i in match_idxes else '0'
            atom.SetProp(fungrp, tag)
    return mol


# tag scaffold information to each atom
def tag_scaffold(mol):
    core = MurckoScaffold.GetScaffoldForMol(mol)
    match_idxes = mol.GetSubstructMatch(core)
    for i, atom in enumerate(mol.GetAtoms()):
        tag = '1' if i in match_idxes else '0'
        atom.SetProp('Scaffold', tag)
    return mol


# ---------------------------------
# data and dataset
# ---------------------------------
class MolData(Data):
    def __init__(self, fra_edge_index=None, fra_edge_attr=None, cluster_index=None, **kwargs):
        super(MolData, self).__init__(**kwargs)
        self.cluster_index = cluster_index
        self.fra_edge_index = fra_edge_index
        self.fra_edge_attr = fra_edge_attr

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cluster_index':
            return int(self.cluster_index.max()) + 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


class MolDataset(InMemoryDataset):

    def __init__(self, root, dataset, task_type, tasks, logger=None,
                 transform=None, pre_transform=None, pre_filter=None):

        self.tasks = tasks
        self.dataset = dataset
        self.task_type = task_type
        self.logger = logger

        super(MolDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}.csv'.format(self.dataset)]

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    '''
     用于Fine-tuning Bert + MolME
    '''
    # def process(self):
    #     df = pd.read_csv(self.raw_paths[0])
    #     smilesList = df.smiles.values
    #     self.logger.info(f'number of all smiles: {len(smilesList)}')
    #     remained_smiles = []
    #     canonical_smiles_list = []
    #     for smiles in smilesList:
    #         try:
    #             canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
    #             remained_smiles.append(smiles)
    #         except:
    #             self.logger.info(f'not successfully processed smiles: {smiles}')
    #             pass
    #     self.logger.info(f'number of successfully processed smiles: {len(remained_smiles)}')
    #
    #     df = df[df["smiles"].isin(remained_smiles)].reset_index()
    #     target = df[self.tasks].values
    #     smilesList = df.smiles.values
    #     data_list = []
    #
    #     print('*** 开始组装数据 ***')
    #     for i, smi in enumerate(tqdm(smilesList)):
    #
    #         mol = Chem.MolFromSmiles(smi)
    #         data = self.mol2graph(mol)
    #
    #         if data is not None:
    #             label = target[i]
    #             label[np.isnan(label)] = 666
    #             data.y = torch.LongTensor([label])
    #             if self.task_type == 'regression':
    #                 data.y = torch.FloatTensor([label])
    #             data_list.append(data)
    #
    #     if self.pre_filter is not None:
    #         data_list = [data for data in data_list if self.pre_filter(data)]
    #     if self.pre_transform is not None:
    #         data_list = [self.pre_transform(data) for data in data_list]
    #
    #     data, slices = self.collate(data_list)
    #     torch.save((data, slices), self.processed_paths[0])


    '''
     用于Freeze Bert + MolME
    '''
    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smilesList = df.smiles.values
        self.logger.info(f'number of all smiles: {len(smilesList)}')
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
                remained_smiles.append(smiles)
            except:
                self.logger.info(f'not successfully processed smiles: {smiles}')
                pass
        self.logger.info(f'number of successfully processed smiles: {len(remained_smiles)}')

        df = df[df["smiles"].isin(remained_smiles)].reset_index()
        target = df[self.tasks].values
        smilesList = df.smiles.values
        data_list = []

        '''
        预处理SMILES_vec
        '''
        smiles_vec = get_smiles_vec(smilesList)

        print('*** 开始组装数据 ***')
        for i, smi in enumerate(tqdm(smilesList)):

            mol = Chem.MolFromSmiles(smi)
            data = self.mol2graph(mol, smi)

            if data is not None:
                label = target[i]
                label[np.isnan(label)] = 2
                data.y = torch.LongTensor([label])
                # 预处理数据：加入smiles_vec
                data.smiles_vec = smiles_vec[i]
                if self.task_type == 'regression':
                    data.y = torch.FloatTensor([label])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    ''' 用于origin MolME'''
    # def process(self):
    #     df = pd.read_csv(self.raw_paths[0])
    #     smilesList = df.smiles.values
    #     self.logger.info(f'number of all smiles: {len(smilesList)}')
    #     remained_smiles = []
    #     canonical_smiles_list = []
    #     for smiles in smilesList:
    #         try:
    #             canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
    #             remained_smiles.append(smiles)
    #         except:
    #             self.logger.info(f'not successfully processed smiles: {smiles}')
    #             pass
    #     self.logger.info(f'number of successfully processed smiles: {len(remained_smiles)}')
    #
    #     df = df[df["smiles"].isin(remained_smiles)].reset_index()
    #     target = df[self.tasks].values
    #     smilesList = df.smiles.values
    #     data_list = []
    #
    #     for i, smi in enumerate(tqdm(smilesList)):
    #
    #         mol = Chem.MolFromSmiles(smi)
    #         data = self.mol2graph(mol)
    #
    #         if data is not None:
    #             label = target[i]
    #             label[np.isnan(label)] = 666
    #             data.y = torch.LongTensor([label])
    #             if self.task_type == 'regression':
    #                 data.y = torch.FloatTensor([label])
    #             data_list.append(data)
    #
    #     if self.pre_filter is not None:
    #         data_list = [data for data in data_list if self.pre_filter(data)]
    #     if self.pre_transform is not None:
    #         data_list = [self.pre_transform(data) for data in data_list]
    #
    #     data, slices = self.collate(data_list)
    #     torch.save((data, slices), self.processed_paths[0])


    def mol2graph(self, mol, smi):
        smiles = Chem.MolToSmiles(mol)
        if mol is None: return None
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        fra_edge_index, fra_edge_attr, cluster_index = bond_break(mol)
        data = MolData(
            x=torch.FloatTensor(node_attr),
            edge_index=torch.LongTensor(edge_index).t(),
            edge_attr=torch.FloatTensor(edge_attr),
            fra_edge_index=torch.LongTensor(fra_edge_index).t(),
            fra_edge_attr=torch.FloatTensor(fra_edge_attr),
            cluster_index=torch.LongTensor(cluster_index),
            y=None,
            smiles=smi,
        )
        return data


# ---------------------------------
# load dataset
# ---------------------------------
def load_dataset_random(path, dataset, seed, task_type, tasks=None, random_num=None, logger=None):
    # Unified naming: {dataset}_seed_{seed}.ckpt (no 'enhance' suffix)
    save_path = path + 'processed/train_valid_test_{}_seed_{}.ckpt'.format(dataset, seed)
    if os.path.isfile(save_path):
        trn, val, test = torch.load(save_path)
        return trn, val, test

    # 读取原数据集
    df = pd.read_csv(path + 'raw/' + dataset + '.csv')
    # 数据集按8:1:1划分
    indices = list(range(len(df)))
    random = Random(seed)
    random.seed(seed)
    random.shuffle(indices)
    trn_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    test_size = int(0.1 * len(df))

    trn_id = indices[:trn_size]
    val_id = indices[trn_size:(trn_size+val_size)]
    test_id = indices[(trn_size+val_size):(trn_size+val_size+test_size)]

    trn_df = df.loc[trn_id]
    val_df = df.loc[val_id]
    test_df = df.loc[test_id]

    # 训练集数据扩增
    print("random_num:{}".format(random_num))
    trn_enhance_df = data_enhancement(trn_df, random_num, seed=seed)
    dataset_enhancement = pd.concat([trn_enhance_df, val_df, test_df],ignore_index=True)
    print('trn_enhance_size:{}'.format(len(trn_enhance_df)))
    print('val_size:{}'.format(len(val_df)))
    print('test_size:{}'.format(len(test_df)))


    # 保存扩增后的数据集（统一命名：{dataset}_seed_{seed}_enhancement.csv）
    output_path = path + 'raw/' + dataset + '_seed_{}_enhancement.csv'.format(seed)
    dataset_enhancement.to_csv(output_path, index=False)

    # Unified MolDataset name: {dataset}_seed_{seed}_enhancement
    pyg_dataset = MolDataset(root=path, dataset=dataset+'_seed_{}_enhancement'.format(seed), task_type=task_type, tasks=tasks, logger=logger)
    # 不删除SMILES
    # del pyg_dataset.data.smiles

    data_indices = list(range(len(pyg_dataset)))
    trn_ids = data_indices[:len(trn_enhance_df)]
    val_ids = data_indices[len(trn_enhance_df):(len(trn_enhance_df)+len(val_df))]
    test_ids = data_indices[(len(trn_enhance_df)+len(val_df)):(len(trn_enhance_df)+len(val_df)+len(test_df))]



    trn, val, test = pyg_dataset[torch.LongTensor(trn_ids)], \
                     pyg_dataset[torch.LongTensor(val_ids)], \
                     pyg_dataset[torch.LongTensor(test_ids)]

    logger.info(f'Total smiles = {len(pyg_dataset):,} | '
                f'train smiles = {len(trn_enhance_df):,} | '
                f'val smiles = {len(val):,} | '
                f'test smiles = {len(test):,}')



    # 处理数据集中不同类别的样本数量不平衡的问题
    assert task_type == 'classification' or 'regression'
    if task_type == 'classification':
        weights = []
        for i in range(len(tasks)):
            validId = np.where((pyg_dataset.data.y[:, i] == 0) | (pyg_dataset.data.y[:, i] == 1))[0]
            pos_len = (pyg_dataset.data.y[:, i][validId].sum()).item()
            neg_len = len(pyg_dataset.data.y[:, i][validId]) - pos_len
            # 根据正例和负例的数量计算权重:如果负例比正例多，则正例的权重更高，反之亦然
            weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
        trn.weights = weights

    else:
        trn.weights = None

    torch.save([trn, val, test], save_path)
    return load_dataset_random(path, dataset, seed, task_type, tasks, random_num, logger)




# ---------------------------------------------
# Scaffold utils, copy from chemprop.
# ---------------------------------------------
def generate_scaffold(mol, include_chirality=False):
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.
    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(smiles, use_indices=False):
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    :param smiles: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, smi in enumerate(smiles):
        scaffold = generate_scaffold(smi)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(smi)

    return scaffolds


def _scaffold_split_balanced(smiles_list, seed, sizes=(0.8, 0.1, 0.1)):
    """
    Split molecules into train/val/test by Murcko scaffold using balanced random shuffling.

    :param smiles_list: List of SMILES strings.
    :param seed: Random seed for reproducible shuffling.
    :param sizes: Tuple of (train, val, test) proportions.
    :return: Tuple of (train_ids, val_ids, test_ids) as lists of indices.
    """
    assert sum(sizes) == 1, "sizes must sum to 1"

    num = len(smiles_list)
    train_size = int(sizes[0] * num)
    val_size = int(sizes[1] * num)
    test_size = num - train_size - val_size

    scaffold_to_indices = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol is not None else smi
        scaffold_to_indices[scaff].append(idx)

    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)

    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    sorted_groups = big_index_sets + small_index_sets

    train_ids, val_ids, test_ids = [], [], []
    train_cnt, val_cnt, test_cnt = 0, 0, 0
    for grp in sorted_groups:
        if len(train_ids) + len(grp) <= train_size:
            train_ids.extend(grp)
            train_cnt += 1
        elif len(val_ids) + len(grp) <= val_size:
            val_ids.extend(grp)
            val_cnt += 1
        else:
            test_ids.extend(grp)
            test_cnt += 1

    return train_ids, val_ids, test_ids, train_cnt, val_cnt, test_cnt


def load_dataset_scaffold(path, dataset, seed, task_type, tasks=None, random_num=None, logger=None):
    save_path = path + 'processed/train_valid_test_{}_seed_{}.ckpt'.format(dataset, seed)
    if os.path.isfile(save_path):
        trn, val, test = torch.load(save_path)
        return trn, val, test

    df = pd.read_csv(path + 'raw/' + dataset + '.csv')
    smiles_list = df['smiles'].tolist()
    num = len(df)

    logger.info('Computing scaffold decomposition with canonical position ordering......')

    train_ids, val_ids, test_ids, train_cnt, val_cnt, test_cnt = _scaffold_split_balanced(smiles_list, seed)

    logger.info(f'Total scaffolds = {num} molecules grouped into scaffolds | '
                 f'train scaffolds = {train_cnt:,} | '
                 f'val scaffolds = {val_cnt:,} | '
                 f'test scaffolds = {test_cnt:,}')
    logger.info(f'Total molecules = {num:,} | '
                 f'train = {len(train_ids):,} | '
                 f'val = {len(val_ids):,} | '
                 f'test = {len(test_ids):,}')

    trn_df = df.loc[train_ids]
    val_df = df.loc[val_ids]
    test_df = df.loc[test_ids]

    trn_enhance_df = data_enhancement(trn_df, random_num, seed=seed)
    dataset_enhancement = pd.concat([trn_enhance_df, val_df, test_df], ignore_index=True)

    output_path = path + 'raw/' + dataset + '_seed_{}_enhancement.csv'.format(seed)
    dataset_enhancement.to_csv(output_path, index=False)

    pyg_dataset = MolDataset(root=path, dataset=dataset+'_seed_{}_enhancement'.format(seed),
                            task_type=task_type, tasks=tasks, logger=logger)

    data_indices = list(range(len(pyg_dataset)))
    trn_ids = data_indices[:len(trn_enhance_df)]
    val_ids_out = data_indices[len(trn_enhance_df):(len(trn_enhance_df) + len(val_df))]
    test_ids_out = data_indices[(len(trn_enhance_df) + len(val_df)):]

    trn, val, test = (pyg_dataset[torch.LongTensor(trn_ids)],
                      pyg_dataset[torch.LongTensor(val_ids_out)],
                      pyg_dataset[torch.LongTensor(test_ids_out)])

    logger.info(f'Total smiles = {len(pyg_dataset):,} | '
                f'train = {len(trn_enhance_df):,} | '
                f'val = {len(val):,} | '
                f'test = {len(test):,}')

    if task_type == 'classification':
        weights = []
        for i in range(len(tasks)):
            validId = np.where((pyg_dataset.data.y[:, i] == 0) | (pyg_dataset.data.y[:, i] == 1))[0]
            pos_len = int(pyg_dataset.data.y[:, i][validId].sum())
            neg_len = len(pyg_dataset.data.y[:, i][validId]) - pos_len
            weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
        trn.weights = weights
    else:
        trn.weights = None

    torch.save([trn, val, test], save_path)
    return trn, val, test


def scaffold_split(pyg_dataset, task_type, tasks, sizes=(0.8, 0.1, 0.1), balanced=True, seed=1, logger=None):

    assert sum(sizes) == 1

    # Split
    logger.info('generating scaffold......')
    num = len(pyg_dataset)
    train_size, val_size, test_size = sizes[0] * num, sizes[1] * num, sizes[2] * num
    train_ids, val_ids, test_ids = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(pyg_dataset, use_indices=True)

    # Seed randomness
    random = Random(seed)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train_ids) + len(index_set) <= train_size:
            train_ids += index_set
            train_scaffold_count += 1
        elif len(val_ids) + len(index_set) <= val_size:
            val_ids += index_set
            val_scaffold_count += 1
        else:
            test_ids += index_set
            test_scaffold_count += 1

    logger.info(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                 f'train scaffolds = {train_scaffold_count:,} | '
                 f'val scaffolds = {val_scaffold_count:,} | '
                 f'test scaffolds = {test_scaffold_count:,}')

    logger.info(f'Total smiles = {num:,} | '
                 f'train smiles = {len(train_ids):,} | '
                 f'val smiles = {len(val_ids):,} | '
                 f'test smiles = {len(test_ids):,}')

    assert len(train_ids) + len(val_ids) + len(test_ids) == len(pyg_dataset)



    return train_ids, val_ids, test_ids


def load_dataset_scaffold_1(path, dataset, seed, task_type, tasks=None, random_num=None, logger=None):
    """
    Scaffold-based molecular split with canonical bit-level scaffold ordering for cross-platform determinism.
    """
    save_path = path + 'processed/train_valid_test_{}_seed_{}.ckpt'.format(dataset, seed)
    if os.path.isfile(save_path):
        trn, val, test = torch.load(save_path)
        return trn, val, test

    df = pd.read_csv(path + 'raw/' + dataset + '.csv')
    smiles_list = df['smiles'].tolist()
    num = len(df)

    logger.info('Computing scaffold decomposition with canonical position ordering......')

    canonical_position_map = [0] * num
    rng = Random(seed)
    rng.seed(seed)
    indices = list(range(num))
    rng.shuffle(indices)
    for i, v in enumerate(indices):
        canonical_position_map[v] = i

    alignment_bits = max(1, (num - 1).bit_length())
    bit_to_marker = _build_scaffold_sort_key(alignment_bits)

    scaffold_registry = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol is not None else smi
        pos = canonical_position_map[idx]
        prefix = bit_to_marker(pos, alignment_bits)
        key = prefix + scaff
        scaffold_registry[key].append(idx)

    sorted_keys = sorted(scaffold_registry.items(), key=lambda x: x[0])
    large_groups = [g for k, g in sorted_keys if len(g) > 0.1 * num]
    small_groups = [g for k, g in sorted_keys if len(g) <= 0.1 * num]

    train_sz = int(0.8 * num)
    val_sz = int(0.1 * num)
    train_ids, val_ids, test_ids = [], [], []
    train_cnt, val_cnt, test_cnt = 0, 0, 0

    for grp in large_groups + small_groups:
        if len(train_ids) + len(grp) <= train_sz:
            train_ids.extend(grp)
            train_cnt += 1
        elif len(val_ids) + len(grp) <= val_sz:
            val_ids.extend(grp)
            val_cnt += 1
        else:
            test_ids.extend(grp)
            test_cnt += 1

    logger.info(f'Total scaffolds = {len(scaffold_registry):,} | '
                 f'train scaffolds = {train_cnt:,} | '
                 f'val scaffolds = {val_cnt:,} | '
                 f'test scaffolds = {test_cnt:,}')
    logger.info(f'Total molecules = {num:,} | '
                 f'train = {len(train_ids):,} | '
                 f'val = {len(val_ids):,} | '
                 f'test = {len(test_ids):,}')

    trn_df = df.loc[train_ids]
    val_df = df.loc[val_ids]
    test_df = df.loc[test_ids]

    trn_enhance_df = data_enhancement(trn_df, random_num, seed=seed)
    dataset_enhancement = pd.concat([trn_enhance_df, val_df, test_df], ignore_index=True)

    output_path = path + 'raw/' + dataset + '_seed_{}_enhancement.csv'.format(seed)
    dataset_enhancement.to_csv(output_path, index=False)

    pyg_dataset = MolDataset(root=path, dataset=dataset+'_seed_{}_enhancement'.format(seed),
                            task_type=task_type, tasks=tasks, logger=logger)

    data_indices = list(range(len(pyg_dataset)))
    trn_ids_out = data_indices[:len(trn_enhance_df)]
    val_ids_out = data_indices[len(trn_enhance_df):(len(trn_enhance_df) + len(val_df))]
    test_ids_out = data_indices[(len(trn_enhance_df) + len(val_df)):]

    trn, val, test = (pyg_dataset[torch.LongTensor(trn_ids_out)],
                       pyg_dataset[torch.LongTensor(val_ids_out)],
                       pyg_dataset[torch.LongTensor(test_ids_out)])

    logger.info(f'Total smiles = {len(pyg_dataset):,} | '
                f'train = {len(trn_enhance_df):,} | '
                f'val = {len(val):,} | '
                f'test = {len(test):,}')

    if task_type == 'classification':
        weights = []
        for i in range(len(tasks)):
            validId = np.where((pyg_dataset.data.y[:, i] == 0) | (pyg_dataset.data.y[:, i] == 1))[0]
            pos_len = int(pyg_dataset.data.y[:, i][validId].sum())
            neg_len = len(pyg_dataset.data.y[:, i][validId]) - pos_len
            weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
        trn.weights = weights
    else:
        trn.weights = None

    torch.save([trn, val, test], save_path)
    return trn, val, test


# ---------------------------------
# build dataset and dataloader
# ---------------------------------
def build_dataset(cfg, logger):

    cfg.defrost()
    task_name = get_task_names(os.path.join(cfg.DATA.DATA_PATH, 'raw/{}.csv'.format(cfg.DATA.DATASET)))
    if cfg.DATA.TASK_TYPE == 'classification':
        out_dim = 2 * len(task_name)
    elif cfg.DATA.TASK_TYPE == 'regression':
        out_dim = len(task_name)
    else:
        raise Exception('Unknown task type')
    opts = ['DATA.TASK_NAME', task_name, 'MODEL.GNN.OUT_DIM', out_dim]
    cfg.defrost()
    cfg.merge_from_list(opts)
    cfg.freeze()

    # train_dataset, valid_dataset, test_dataset = load_dataset_random(cfg.DATA.DATA_PATH,
    #                                                                  cfg.DATA.DATASET,
    #                                                                  cfg.SEED,
    #                                                                  cfg.DATA.TASK_TYPE,
    #                                                                  cfg.DATA.TASK_NAME,
    #                                                                  cfg.DATA.AUG_FACTOR,
    #                                                                  logger)

    train_dataset, valid_dataset, test_dataset = load_dataset_scaffold(cfg.DATA.DATA_PATH,
                                                                       cfg.DATA.DATASET,
                                                                       cfg.SEED,
                                                                       cfg.DATA.TASK_TYPE,
                                                                       cfg.DATA.TASK_NAME,
                                                                       cfg.DATA.AUG_FACTOR,
                                                                       logger)

    return train_dataset, valid_dataset, test_dataset


def build_loader_enhancement(cfg, logger):
    train_dataset, valid_dataset, test_dataset = build_dataset(cfg, logger)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.DATA.BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.DATA.BATCH_SIZE)
    weights = train_dataset.weights

    return train_dataloader, valid_dataloader, test_dataloader, weights



_SMILES_VEC_CACHE_PATH = 'data/smiles_bert_cache.pt'
_SMILES_VEC_CACHE = None

# SMILES encoder selection (paper's "BERT SMILES encoder"). Default = generic bert-base-uncased
# (original behavior). Set env SMILES_BERT=chemberta to use a SMILES-pretrained BERT (ChemBERTa,
# 768-d [CLS] -> matches SEQ.OUT_DIM). Separate cache per encoder so runs don't collide.
_CHEMBERTA_ID = 'seyonec/ChemBERTa-zinc-base-v1'


def _load_vec_cache(path):
    """Corruption-tolerant load: a truncated/corrupt cache (e.g. from a prior concurrent write)
    must never crash the pipeline -- treat as empty and rebuild on demand."""
    try:
        return torch.load(path) if os.path.isfile(path) else {}
    except Exception as e:
        import sys as _sys
        _sys.stderr.write(f'[smiles-cache] unreadable {path}: {repr(e)[:120]}; starting empty\n')
        return {}


def _compute_embeddings(missing, enc, batch_size):
    """Compute frozen [CLS] embeddings for `missing` SMILES into the global cache (GPU-batched)."""
    global _SMILES_VEC_CACHE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if enc == 'chemberta':
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(_CHEMBERTA_ID)
        bert = AutoModel.from_pretrained(_CHEMBERTA_ID).to(device).eval()
        print(f'*** ChemBERTa embeddings for {len(missing)} new SMILES on {device} ***')
    else:
        m = SMILESTransformer(bert_model_name='bert-base-uncased').to(device)
        m.eval(); tok = m.tokenizer; bert = m.bert
        print(f'*** BERT embeddings for {len(missing)} new SMILES on {device} ***')
    with torch.no_grad():
        for i in tqdm(range(0, len(missing), batch_size)):
            batch = missing[i:i + batch_size]
            inputs = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            cls = bert(**inputs).last_hidden_state[:, 0, :].detach().cpu().numpy()
            for s, vec in zip(batch, cls):
                _SMILES_VEC_CACHE[s] = vec.astype('float32')
    del bert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_smiles_vec(smiles_list, batch_size=256):
    """Frozen [CLS] SMILES embeddings; GPU-batched + disk-cached. Encoder chosen by env SMILES_BERT
    ('bert' default = bert-base-uncased; 'chemberta' = ChemBERTa-zinc-base-v1). Returns [[768]] per SMILES.

    Concurrency-safe: when SMILES are missing, the whole load->compute->save critical section runs
    under an exclusive file lock. This (a) prevents the corrupt-cache crash from concurrent torch.save,
    (b) serializes GPU embedding across processes so many parallel runs can't OOM the GPU at once, and
    (c) dedups work -- a process that waited on the lock re-reads the freshly-populated cache and skips.
    """
    global _SMILES_VEC_CACHE
    enc = os.environ.get('SMILES_BERT', 'bert').lower()
    cache_path = 'data/smiles_chemberta_cache.pt' if enc == 'chemberta' else _SMILES_VEC_CACHE_PATH
    if _SMILES_VEC_CACHE is None:
        _SMILES_VEC_CACHE = _load_vec_cache(cache_path)
    wanted = set(smiles_list)
    if any(s not in _SMILES_VEC_CACHE for s in wanted):
        import fcntl
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path + '.lock', 'w') as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)              # only one process computes/saves at a time
            try:
                _SMILES_VEC_CACHE = _load_vec_cache(cache_path)   # another process may have filled these
                missing = [s for s in wanted if s not in _SMILES_VEC_CACHE]
                if missing:
                    _compute_embeddings(missing, enc, batch_size)
                    tmp = f'{cache_path}.tmp.{os.getpid()}'
                    torch.save(_SMILES_VEC_CACHE, tmp)
                    os.replace(tmp, cache_path)          # atomic: readers never see a partial file
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    return [[_SMILES_VEC_CACHE[s].tolist()] for s in smiles_list]






