import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import DataLoader

def edge_index_to_adjacency(edge_index, num_nodes=None):
    """
    将边索引转换为邻接矩阵。
    """
    edge_index = np.array(edge_index)
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    adjacency[edge_index[0, :], edge_index[1, :]] = 1
    # adjacency += adjacency.T  # 对称矩阵
    return adjacency
 
def adjacency_to_edge_index(adjacency):
    """
    将邻接矩阵转换为边索引。
    """
    adjacency = np.array(adjacency)
    edge_index = np.array(np.nonzero(adjacency))
    return edge_index

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element of the permitted list.
    将x转为onehot编码
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    # define list of permitted atoms
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    # compute atom features
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms) # length:43,获取原子符号
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])  # length:6,原子度，该原子形成键的数目
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"]) # length:8,获取原子的电荷
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]) # length:7,获取原子杂化方式
    is_in_a_ring_enc = [int(atom.IsInRing())]  # length：1，原子是在环上
    is_aromatic_enc = [int(atom.GetIsAromatic())]  # length：1，原子是否是芳香原子
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]  # length：1，相对原子质量
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]  # length：1，范德华半径
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)] # # length：1，共价键半径
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled                                
    if use_chirality == True: # length:4,原子手性
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    if hydrogens_implicit == True: # length:6,
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)

def get_bond_features(bond, use_stereochemistry = True):
    """
    获取键特征，Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types) # 键的类型，单键，双键，三键，芳香键，采用独热编码
    bond_is_conj_enc = [int(bond.GetIsConjugated())] # 是否共轭
    bond_is_in_ring_enc = [int(bond.IsInRing())] # 是否在环上
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]) # 手性信息，独热编码
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    Inputs:
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    Outputs:
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    """
    data_list = []
    for (smiles, y_val) in zip(x_smiles, y):
        # 便利smiles和标签
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles) # 读取smiles，返回mol对象，每个mol对象保存
        # get feature dimensions
        n_nodes = mol.GetNumAtoms() # 返回原子数量作为图的节点数
        n_edges = 2*mol.GetNumBonds() # 返回边的数量
        unrelated_smiles = "O=O" # 用氧分子作为例子，定义节点和边的特征向量长度
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom) # 获取每个原子的特征
        X = torch.tensor(X, dtype = torch.float)
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol)) # 返回原子的邻接矩阵，表示相邻的原子
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0) # 边的连接关系
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features)) # 建立边特征
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            # mol.GetBondBetweenAtoms(int(i),int(j)) 返回节点i和节点j的边对象
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        EF = torch.tensor(EF, dtype = torch.float)
        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float) # 处理标签
        # construct Pytorch Geometric data object and append to data list
        # 节点属性，边，边属性，label
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))
    return data_list

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels2(x_smiles, x_sizes=None):
    """
    Inputs:
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    Outputs:
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    """
    node_ls = []
    edge_index_ls = []
    edge_attr_ls = []
    edge_adj_ls = []
    for i, smiles in enumerate(x_smiles):
        # 便利smiles和标签
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles) # 读取smiles，返回mol对象，每个mol对象保存
        # get feature dimensions
        n_nodes = mol.GetNumAtoms() # 返回原子数量作为图的节点数
        n_edges = 2*mol.GetNumBonds() # 返回边的数量
        unrelated_smiles = "O=O" # 用氧分子作为例子，定义节点和边的特征向量长度
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
        X = np.zeros((n_nodes, n_node_features))
        # construct node feature matrix X of shape (n_nodes, n_node_features)
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom) # 获取每个原子的特征
        if x_sizes is not None:
            x_size = x_sizes[i]
            n_node_features += 1
            tmp_X = np.zeros((n_nodes, n_node_features))
            tmp_X[:,:-1] = X
            tmp_X[:,-1] = x_size # 所有原子的粒径属性相同
            X = tmp_X
        X = torch.tensor(X, dtype = torch.float)
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol)) # 返回原子的邻接矩阵，表示相邻的原子
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0) # 边的连接关系
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features)) # 建立边特征
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            # mol.GetBondBetweenAtoms(int(i),int(j)) 返回节点i和节点j的边对象
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        EF = torch.tensor(EF, dtype = torch.float) 
        node_ls.append(X)
        edge_index_ls.append(E)
        edge_attr_ls.append(EF)
        edge_adj_ls.append(edge_index_to_adjacency(E, X.shape[0]))
    return  node_ls, edge_index_ls, edge_attr_ls, edge_adj_ls

class GNNDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(GNNDataset, self).__init__()
        x_smiles = pd.read_csv('dataset/selected_drugs_smiles1.tsv', sep='\t')
        smiles_ls = list(x_smiles.iloc[:,-1])
        y = [0] * len(smiles_ls)
        self.data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles_ls, y)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        obj = self.data_list[index]
        return obj.x, obj.edge_index, obj.edge_attr, obj.y
