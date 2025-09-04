import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import argparse
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from conv import GNN_node, GNN_node_Virtualnode

from torch_geometric.nn import MessagePassing

from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import degree
from torch_geometric.utils import softmax
from torch_geometric.nn.norm import GraphNorm
import math

from ogb.utils.features import bond_to_feature_vector, atom_to_feature_vector


from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import pathlib
import os.path as osp
import pandas as pd
import numpy as np
import torch
import copy

DATA_DIR = "/kaggle/input/neurips-open-polymer-prediction-2025"
TRAIN_FILE_NAME = "train.csv"
TARGET = "Tg"

import torch
import argparse
from sklearn.metrics import r2_score


def get_args():
    parser = argparse.ArgumentParser(
        description="Graph rationalization with Environment-based Augmentation"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    # model
    parser.add_argument(
        "--gnn",
        type=str,
        default="gin-virtual",
        help="GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)",
    )
    parser.add_argument(
        "--drop_ratio", type=float, default=0.5, help="dropout ratio (default: 0.5)"
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5)",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=128,
        help="dimensionality of hidden units in GNNs (default: 128)",
    )
    parser.add_argument(
        "--use_linear_predictor",
        default=False,
        action="store_true",
        help="Use Linear predictor",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.4,
        help="size ratio to regularize the rationale subgraph (default: 0.4)",
    )

    # training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="input batch size for training (default: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="patience for early stop (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Learning rate (default: 1e-2)"
    )
    parser.add_argument(
        "--l2reg", type=float, default=5e-6, help="L2 norm (default: 5e-6)"
    )
    parser.add_argument(
        "--use_lr_scheduler",
        default=False,
        action="store_true",
        help="Use learning rate scheduler CosineAnnealingLR",
    )
    parser.add_argument(
        "--use_clip_norm",
        default=False,
        action="store_true",
        help="Use learning rate clip norm",
    )
    parser.add_argument(
        "--path_list",
        nargs="+",
        default=[1, 4],
        help="path for alternative optimization",
    )
    parser.add_argument(
        "--initw_name",
        type=str,
        default="default",
        choices=["default", "orthogonal", "normal", "xavier", "kaiming"],
        help="method name to initialize neural weights",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-molbbbp",
        help="dataset name (default: ogbg-molhiv)",
    )
    parser.add_argument(
        "--trails", type=int, default=5, help="numer of experiments (default: 5)"
    )
    parser.add_argument(
        "--by_default",
        default=False,
        action="store_true",
        help="use default configuration for hyperparameters",
    )
    args = parser.parse_args()

    return args


cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(args, model, device, loader, optimizers, task_type, optimizer_name):
    optimizer = optimizers[optimizer_name]
    model.train()
    if optimizer_name == "predictor":
        set_requires_grad([model.graph_encoder, model.predictor], requires_grad=True)
        set_requires_grad([model.separator], requires_grad=False)
    if optimizer_name == "separator":
        set_requires_grad([model.separator], requires_grad=True)
        set_requires_grad([model.graph_encoder, model.predictor], requires_grad=False)

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            pred = model(batch)
            if "classification" in task_type:
                criterion = cls_criterion
            else:
                criterion = reg_criterion

            if args.dataset.startswith("plym"):
                if args.plym_prop == "density":
                    batch.y = torch.log(batch[args.plym_prop])
                else:
                    batch.y = batch[args.plym_prop]
            target = batch.y.to(torch.float32)
            is_labeled = batch.y == batch.y
            loss = criterion(
                pred["pred_rem"].to(torch.float32)[is_labeled], target[is_labeled]
            )
            target_rep = batch.y.to(torch.float32).repeat_interleave(
                batch.batch[-1] + 1, dim=0
            )
            is_labeled_rep = target_rep == target_rep
            loss += criterion(
                pred["pred_rep"].to(torch.float32)[is_labeled_rep],
                target_rep[is_labeled_rep],
            )

            if optimizer_name == "separator":
                loss += pred["loss_reg"]
            # ここで学習
            loss.backward()
            if args.use_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def eval(args, model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model.eval_forward(batch)

            if args.dataset.startswith("plym"):
                if args.plym_prop == "density":
                    batch.y = torch.log(batch[args.plym_prop])
                else:
                    batch.y = batch[args.plym_prop]
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    if args.dataset.startswith("plym"):
        return [evaluator.eval(input_dict)["rmse"], r2_score(y_true, y_pred)]
    elif args.dataset.startswith("ogbg"):
        return [evaluator.eval(input_dict)["rocauc"]]


def generate(args, model, device, loader):
    model.eval()
    graph_chunks = []
    target_chunks = []
    total_nodes = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        n_nodes = batch.x.shape[0]
        print(f"[generate] batch={step} nodes={n_nodes}")
        if n_nodes <= 1:
            continue
        with torch.no_grad():
            h_rep, target_rep = model.generate_graph(batch)
            graph_chunks.append(h_rep)
            target_chunks.append(target_rep)
            total_nodes += h_rep.size(0)

    if len(graph_chunks) == 0:
        return {"graph": torch.empty(0, model.emb_dim), "y": torch.empty(0, 1)}

    graphs_cat = torch.cat(graph_chunks, dim=0)
    targets_cat = torch.cat(target_chunks, dim=0)
    print(f"[generate] concatenated: {graphs_cat.shape}  targets: {targets_cat.shape}")
    return {"graph": graphs_cat, "y": targets_cat}


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == "default":
                pass
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class PolymerRegDataset(InMemoryDataset):
    def __init__(self, name="o2_prop", root="data", transform=None, pre_transform=None):
        """
        - name (str): name of the dataset
        - root (str): root directory to store the dataset folder
        - transform, pre_transform (optional): transform/pre-transform graph objects
        """
        self.name = name
        self.original_root = root
        self.processed_root = osp.join(osp.dirname(osp.abspath(root)))

        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.task_type = "regression"
        self.__num_classes__ = "-1"
        self.binary = "False"

        super(PolymerRegDataset, self).__init__(
            self.processed_root, transform, pre_transform
        )

        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def process(self):
        print("process")
        read_path = osp.join(self.original_root, self.name)
        data_list = self.read_graph_pyg(read_path)
        print(data_list[:3])
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def csv2graphs(self, raw_dir):
        """
        - raw_dir: the position where gas property csv stored,
        the name of the file is the gas name,
        each file contains two columns: one for smiles, one for property value
        """
        dfs = []
        path_suffix = pathlib.Path(raw_dir).suffix
        if path_suffix == "":  # is path
            for file_name in os.listdir(raw_dir):
                if len(file_name) <= 10:
                    df_temp = pd.read_csv(
                        "{}/{}".format(raw_dir, file_name), engine="python"
                    )
                    df_temp.set_index("SMILES", inplace=True)
                    dfs.append(df_temp)
                    print(file_name, ":", len(df_temp.index))
            df_full = pd.concat(dfs).groupby(level=0).mean().fillna(-1)
        elif path_suffix == ".csv":
            df_full = pd.read_csv(raw_dir, engine="python")
            df_full.set_index("SMILES", inplace=True)
            print(df_full[:5])

        target_col = TARGET
        print("sssss")
        if target_col in df_full.columns:
            before = len(df_full)
            print(df_full.columns)
            df_full = df_full[np.isfinite(df_full[target_col])]
            print(df_full.shape)
            df_full = df_full.dropna(subset=[target_col])
            after = len(df_full)
            if before != after:
                print(
                    f"[clean] Dropped {before} -> {after} rows with invalid {target_col}"
                )
        graph_list = []

        # TODO:ここに各要素を指定する
        for smiles_idx in df_full.index[:]:
            graph_dict = smiles2graph(smiles_idx)
            props = df_full.loc[smiles_idx]
            for name, value in props.items():
                graph_dict[name] = np.array([[value]])
            graph_list.append(graph_dict)
        return graph_list

    def read_graph_pyg(self, raw_dir):
        print("raw_dir", raw_dir)
        graph_list = self.csv2graphs(raw_dir)
        pyg_graph_list = []
        print("Converting graphs into PyG objects...")
        print(type(graph_list))
        for graph in tqdm(graph_list):
            g = Data()
            g.__num_nodes__ = graph["num_nodes"]
            g.edge_index = torch.from_numpy(graph["edge_index"])

            del graph["num_nodes"]
            del graph["edge_index"]

            if graph["edge_feat"] is not None:
                g.edge_attr = torch.from_numpy(graph["edge_feat"])
                del graph["edge_feat"]

            if graph["node_feat"] is not None:
                g.x = torch.from_numpy(graph["node_feat"])
                del graph["node_feat"]

            addition_prop = copy.deepcopy(graph)
            for key in addition_prop.keys():
                g[key] = torch.tensor(graph[key])
                del graph[key]

            pyg_graph_list.append(g)

        return pyg_graph_list


def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    atom_label = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
        atom_label.append(atom.GetSymbol())

    x = np.array(atom_features_list, dtype=np.int64)
    atom_label = np.array(atom_label, dtype=str)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)
    return graph



allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_formal_charge_list': [-5,-4,-3,-2,-1,0,1,2,3,4,5],
    'possible_number_radical_e_list': [0,1,2,3,4],
    'possible_bond_type_list': [
        Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE, Chem.BondType.AROMATIC
    ],
    'possible_bond_stereo_list': [
        Chem.BondStereo.STEREONONE,
        Chem.BondStereo.STEREOZ,
        Chem.BondStereo.STEREOE,
        Chem.BondStereo.STEREOANY
    ],
    'possible_is_conjugated_list': [False, True],
}

def _decode(lst, idx, default):
    return lst[idx] if 0 <= idx < len(lst) else default

def decode_atom_feature(row):
    r = np.asarray(row)
    atomic_idx = int(r[0])
    # 未知 = dummy(*) 扱い
    if atomic_idx == len(allowable_features['possible_atomic_num_list']):
        atomic_num = 0
    else:
        atomic_num = _decode(allowable_features['possible_atomic_num_list'], atomic_idx, 6)
    chiral = _decode(allowable_features['possible_chirality_list'], int(r[1]),
                     Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
    formal = _decode(allowable_features['possible_formal_charge_list'], int(r[3]), 0)
    radical = _decode(allowable_features['possible_number_radical_e_list'], int(r[5]), 0)
    aromatic = (int(r[7]) == 1)
    return dict(atomic_num=atomic_num, chiral=chiral,
                formal=formal, radical=radical, aromatic=aromatic)

def decode_bond_feature(row):
    r = np.asarray(row)
    bt = _decode(allowable_features['possible_bond_type_list'], int(r[0]), Chem.BondType.SINGLE)
    st = _decode(allowable_features['possible_bond_stereo_list'], int(r[1]), Chem.BondStereo.STEREONONE)
    cj = _decode(allowable_features['possible_is_conjugated_list'], int(r[2]), False)
    return bt, st, cj

def _partial_sanitize(mol):
    ops = (Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
           Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
           Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
           Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
           Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    try:
        Chem.SanitizeMol(mol, sanitizeOps=ops)
    except Exception:
        pass

def _normalize_implicit_hs(mol):
    # 明示Hを削除可能な形に調整
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetNoImplicit(True)
            continue
        # 余計な explicit H を一旦 0 に戻し再計算を許可
        if a.GetNumExplicitHs() > 0:
            a.SetNumExplicitHs(0)
        a.SetNoImplicit(False)
        a.UpdatePropertyCache(strict=False)
    # 暗黙H再調整
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    except Exception:
        pass
    # Remove explicit H
    mol2 = Chem.RemoveHs(mol, sanitize=False)
    return mol2

def graph2smiles(graph,
                 canonical=False,
                 sanitize_mode="partial",  # "none" | "partial"
                 isomeric=True):
    edge_index = graph["edge_index"]
    edge_feat  = graph["edge_feat"]
    node_feat  = graph["node_feat"]
    n = graph["num_nodes"]

    rw = Chem.RWMol()

    # 原子追加 (hybridization/implicitH 推定は sanitize に任せる)
    for i in range(n):
        attr = decode_atom_feature(node_feat[i])
        if attr['atomic_num'] == 0:
            atom = Chem.Atom("*")
            atom.SetNoImplicit(True)
        else:
            atom = Chem.Atom(attr['atomic_num'])
            atom.SetFormalCharge(attr['formal'])
            if attr['radical'] > 0:
                atom.SetNumRadicalElectrons(attr['radical'])
            atom.SetChiralTag(attr['chiral'])
            if attr['aromatic']:
                atom.SetIsAromatic(True)
            atom.SetNoImplicit(False)
        rw.AddAtom(atom)

    # ボンド（双方向重複排除）
    added = set()
    E = edge_index.shape[1]
    for k in range(E):
        u = int(edge_index[0, k]); v = int(edge_index[1, k])
        if u == v: continue
        key = (u, v) if u < v else (v, u)
        if key in added: continue
        bt, st, cj = decode_bond_feature(edge_feat[k])
        rw.AddBond(u, v, bt)
        b = rw.GetBondBetweenAtoms(u, v)
        if b:
            if bt == Chem.BondType.AROMATIC:
                b.SetIsAromatic(True)
                for aidx in (u, v):
                    a = rw.GetAtomWithIdx(aidx)
                    if a.GetAtomicNum() != 0:
                        a.SetIsAromatic(True)
            if cj:
                b.SetIsConjugated(True)
            if st != Chem.BondStereo.STEREONONE:
                b.SetStereo(st)
        added.add(key)

    mol = rw.GetMol()

    if sanitize_mode == "partial":
        _partial_sanitize(mol)

    mol = _normalize_implicit_hs(mol)

    smiles = Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)
    return smiles

nn_act = torch.nn.ReLU()  # ReLU()
F_act = F.relu


# MessagePassing = グラフ間のメッセージ伝達を行うための基底クラス
# CNNのConv2dに相当
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GINConv, self).__init__(aggr="add")

        # Dense（全結合に対応？）
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            nn_act,
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        # edge_embedding = 結合ごとの性質を表す学習中に更新される数列。
        out = self.mlp(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )

        return out

    def message(self, x_j, edge_attr):
        return F_act(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr="add")

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm
        ) + F_act(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F_act(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self,
        num_layer,
        emb_dim,
        drop_ratio=0.5,
        JK="last",
        residual=False,
        gnn_name="gin",
    ):
        """
        emb_dim (int): node embedding dimensionality
        num_layer (int): number of GNN message passing layers

        """

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_name == "gin":
                self.convs.append(GINConv(emb_dim))
            elif gnn_name == "gcn":
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError("Undefined GNN type called {}".format(gnn_name))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            # self.batch_norms.append(GraphNorm(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )

        ### computing input node embedding

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F_act(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(
        self,
        num_layer,
        emb_dim,
        drop_ratio=0.5,
        JK="last",
        residual=False,
        gnn_name="gin",
        atom_encode=True,
    ):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encode = atom_encode
        if self.atom_encode:
            self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_name == "gin":
                self.convs.append(GINConv(emb_dim))
            elif gnn_name == "gcn":
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError("Undefined GNN type called {}".format(gnn_name))

            # self.batch_norms.append(GraphNorm(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    nn_act,
                    torch.nn.Linear(2 * emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    nn_act,
                )
            )

    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
        )
        if self.atom_encode:
            h_list = [self.atom_encoder(x)]
        else:
            h_list = [x]

        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F_act(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = (
                    global_add_pool(h_list[layer], batch) + virtualnode_embedding
                )
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training,
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training,
                    )

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


nn_act = torch.nn.ReLU()
F_act = F.relu


class GraphEnvAug(torch.nn.Module):

    def __init__(
        self,
        num_tasks,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        drop_ratio=0.5,
        gamma=0.4,
        use_linear_predictor=False,
        external_only=True,
    ):
        """
        num_tasks (int): number of labels to be predicted
        """

        super(GraphEnvAug, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.gamma = gamma
        self.external_only = external_only

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        gnn_name = gnn_type.split("-")[0]
        emb_dim_rat = emb_dim
        if "virtual" in gnn_type:
            rationale_gnn_node = GNN_node_Virtualnode(
                2,
                emb_dim_rat,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
            )
            self.graph_encoder = GNN_node_Virtualnode(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
            )
        else:
            rationale_gnn_node = GNN_node(
                2,
                emb_dim_rat,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
            )
            self.graph_encoder = GNN_node(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
            )
        self.separator = separator(
            rationale_gnn_node=rationale_gnn_node,
            gate_nn=torch.nn.Sequential(
                torch.nn.Linear(emb_dim_rat, 2 * emb_dim_rat),
                torch.nn.BatchNorm1d(2 * emb_dim_rat),
                nn_act,
                torch.nn.Dropout(),
                torch.nn.Linear(2 * emb_dim_rat, 1),
            ),
            nn=None,
        )
        rep_dim = emb_dim
        if use_linear_predictor:
            self.predictor = torch.nn.Linear(rep_dim, self.num_tasks)
        else:
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(rep_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                nn_act,
                torch.nn.Dropout(),
                torch.nn.Linear(2 * emb_dim, self.num_tasks),
            )

    def forward(self, batched_data):
        print(batched_data)
        h_node = self.graph_encoder(batched_data)
        h_r, h_env, r_node_num, env_node_num = self.separator(batched_data, h_node)
        # グラフ拡張
        # h_r.unsqueeze(1): (G, 1, D) + h_env.unsqueeze(0): (1, G, D) view(-1, self.emb_dim):(G, G, D) → (G*G, D)
        h_rep = (h_r.unsqueeze(1) + h_env.unsqueeze(0)).view(-1, self.emb_dim)
        pred_rem = self.predictor(h_r)
        pred_rep = self.predictor(h_rep)
        loss_reg = torch.abs(
            r_node_num / (r_node_num + env_node_num)
            - self.gamma * torch.ones_like(r_node_num)
        ).mean()
        output = {"pred_rep": pred_rep, "pred_rem": pred_rem, "loss_reg": loss_reg}
        return output

    def eval_forward(self, batched_data):
        h_node = self.graph_encoder(batched_data)
        h_r, _, _, _ = self.separator(batched_data, h_node)
        pred_rem = self.predictor(h_r)
        return pred_rem

    def generate_graph(self, batched_data):
        h_node = self.graph_encoder(batched_data)
        h_r, h_env, r_node_num, env_node_num = self.separator(batched_data, h_node)
        G = h_r.size(0)
        h_rep = (h_r.unsqueeze(1) + h_env.unsqueeze(0)).view(-1, self.emb_dim)

        if args.dataset.startswith("plym"):
            if args.plym_prop == "density":
                y = torch.log(batched_data[args.plym_prop])
            else:
                y = batched_data[args.plym_prop]
        # y 次元を (G, y_dim) に整形
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        # repeat_interleave で各 i の y[i] を G 回並べる → h_rep 並びと一致
        target_rep = y.repeat_interleave(G, dim=0)

        return h_rep, target_rep


class separator(torch.nn.Module):
    def __init__(self, rationale_gnn_node, gate_nn, nn=None):
        super(separator, self).__init__()
        self.rationale_gnn_node = rationale_gnn_node
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.rationale_gnn_node)
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, batched_data, h_node, size=None):
        x = self.rationale_gnn_node(batched_data)
        batch = batched_data.batch
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        h_node = self.nn(h_node) if self.nn is not None else h_node
        assert gate.dim() == h_node.dim() and gate.size(0) == h_node.size(0)
        gate = torch.sigmoid(gate)

        # 出力テンソル形状の決定（h_node の次元に依存）
        out_shape = (size,) + tuple(h_node.shape[1:])  # h_node.dim()==1 の時は (size,)
        device = h_node.device
        dtype = h_node.dtype

        h_out = h_node.new_zeros(out_shape)
        c_out = h_node.new_zeros(out_shape)

        # batch は long 型であること（なければ変換）
        if batch.dtype != torch.long:
            batch = batch.long()

        h_out.index_add_(0, batch, gate * h_node)
        c_out.index_add_(0, batch, (1.0 - gate) * h_node)

        # r_node_num / env_node_num は (size, 1) にしておく（元の動作に合わせる）
        r_node_num = gate.new_zeros((size, gate.size(1)))
        env_node_num = gate.new_zeros((size, gate.size(1)))

        r_node_num.index_add_(0, batch, gate)
        env_node_num.index_add_(0, batch, (1.0 - gate))

        return h_out, c_out, r_node_num + 1e-8, env_node_num + 1e-8


import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import numpy as np
from tqdm import tqdm

## dataset
from sklearn.model_selection import train_test_split
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from pprint import pformat


def main(args):
    print(args)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if args.dataset.startswith("ogbg"):
        dataset = PygGraphPropPredDataset(name=args.dataset, root="data")

        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(
            dataset[split_idx["train"]],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        valid_loader = DataLoader(
            dataset[split_idx["valid"]],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            dataset[split_idx["test"]],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        all_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        evaluator = Evaluator(args.dataset)

    elif args.dataset.startswith("plym"):
        dataset = PolymerRegDataset(
            name=TRAIN_FILE_NAME, root=DATA_DIR
        )  # PolymerRegDataset
        full_idx = list(range(len(dataset)))
        train_ratio = 0.6
        valid_ratio = 0.1
        test_ratio = 0.3
        train_index, test_index, _, _ = train_test_split(
            full_idx, full_idx, test_size=test_ratio, random_state=42
        )
        train_index, val_index, _, _ = train_test_split(
            train_index,
            train_index,
            test_size=valid_ratio / (valid_ratio + train_ratio),
            random_state=42,
        )

        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)

        train_loader = DataLoader(
            dataset[train_index],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        valid_loader = DataLoader(
            dataset[val_index], batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            dataset[test_index],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        all_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        evaluator = Evaluator("ogbg-molesol")  # RMSE metric
    n_train_data, n_val_data, n_test_data = (
        len(train_loader.dataset),
        len(valid_loader.dataset),
        float(len(test_loader.dataset)),
    )
    print(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")

    model = GraphEnvAug(
        gnn_type=args.gnn,
        num_tasks=dataset.num_tasks,
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
        gamma=args.gamma,
        use_linear_predictor=args.use_linear_predictor,
    ).to(device)
    init_weights(model, args.initw_name, init_gain=0.02)
    opt_separator = optim.Adam(
        model.separator.parameters(), lr=args.lr, weight_decay=args.l2reg
    )
    opt_predictor = optim.Adam(
        list(model.graph_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
        weight_decay=args.l2reg,
    )
    optimizers = {"separator": opt_separator, "predictor": opt_predictor}
    if args.use_lr_scheduler:
        schedulers = {}
        for opt_name, opt in optimizers.items():
            schedulers[opt_name] = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=100, eta_min=1e-4
            )
    else:
        schedulers = None
    cnt_wait = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        print("=====Epoch {}".format(epoch))
        path = epoch % int(args.path_list[-1])
        if path in list(range(int(args.path_list[0]))):
            optimizer_name = "separator"
        elif path in list(range(int(args.path_list[0]), int(args.path_list[1]))):
            optimizer_name = "predictor"

        train(
            args,
            model,
            device,
            train_loader,
            optimizers,
            dataset.task_type,
            optimizer_name,
        )

        if schedulers != None:
            schedulers[optimizer_name].step()
        train_perf = eval(args, model, device, train_loader, evaluator)[0]
        valid_perf = eval(args, model, device, valid_loader, evaluator)[0]
        update_test = False
        if epoch != 0:
            if "classification" in dataset.task_type and valid_perf > best_valid_perf:
                update_test = True
            elif (
                "classification" not in dataset.task_type
                and valid_perf < best_valid_perf
            ):
                update_test = True
        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            cnt_wait = 0
            best_epoch = epoch
            test_perfs = eval(args, model, device, test_loader, evaluator)
            if args.dataset.startswith("ogbg"):
                test_auc = test_perfs[0]
                print(
                    {
                        "Metric": "AUC",
                        "Train": train_perf,
                        "Validation": valid_perf,
                        "Test": test_auc,
                    }
                )
            else:
                test_rmse, test_r2 = test_perfs[0], test_perfs[1]
                print(
                    {
                        "Metric": "RMSE",
                        "Train": train_perf,
                        "Validation": valid_perf,
                        "Test": test_rmse,
                        "Test R2": test_r2,
                    }
                )
        else:
            print({"Train": train_perf, "Validation": valid_perf})
            cnt_wait += 1
            if cnt_wait > args.patience:
                break
    print(
        "Finished training! Results from epoch {} with best validation {}.".format(
            best_epoch, best_valid_perf
        )
    )
    graphs = generate(args, model, device, all_loader)
    print("Generated {} graphs.".format(len(graphs["graph"])))
    print("Generated {} graphs.".format(len(graphs["y"])))

    for i in graphs["graph"]:
        print(i)
        smi = graph2smiles(i)
        # print(smi)
        graphs["graph"][i] = smi

    # CSV 出力 (埋め込み + target)
    from pathlib import Path
    import pandas as pd

    if graphs["graph"].numel() == 0:
        print("No graph embeddings to save.")
    else:
        out_dir = Path("tmp")
        out_dir.mkdir(exist_ok=True, parents=True)
        emb = graphs["graph"].detach().cpu().numpy()
        tgt = graphs["y"].detach().cpu().numpy()
        # tgt は (N, 1) 想定
        df = pd.DataFrame(emb)
        df.insert(0, "target", tgt.reshape(-1))
        out_path = out_dir / "graph_embeddings.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved embeddings to {out_path} shape={df.shape}")

    if args.dataset.startswith("ogbg"):
        print("Test auc: {}".format(test_auc))
        return [best_valid_perf, test_auc]
    if args.dataset.startswith("plym"):
        print("Test rmse: {}, Test r2: {} \n".format(test_rmse, test_r2))
        return [best_valid_perf, test_rmse, test_r2]


def config_and_run(args):
    print(args.by_default, args.dataset)
    if args.by_default:
        if args.dataset == "plym-o2_prop":
            # oxygen permeability
            args.gamma = 0.2
            args.epochs = 400
            args.num_layer = 3
            args.drop_ratio = 0.1
            args.batch_size = 32
            args.l2reg = 1e-4
            args.lr = 1e-2
            if args.gnn == "gcn-virtual":
                args.lr = 1e-3
                args.l2reg = 1e-5
                args.patience = 100
        if args.dataset == "plym-mt_prop":
            # melting temperature
            args.epochs = 1#400
            args.l2reg = 1e-5
            args.gamma = 0.05
            args.num_layer = 3
            args.drop_ratio = 0.1
            args.batch_size = 32
            args.lr = 1e-2
            if args.gnn == "gcn-virtual":
                args.lr = 1e-3
            args.patience = 50
        if args.dataset == "plym-tg_prop":
            # glass temperature
            args.epochs = 1#400
            args.l2reg = 1e-5
            args.gamma = 0.05
            args.num_layer = 3
            args.drop_ratio = 0.1
            args.initw_name = "orthogonal"
            args.batch_size = 256
            args.lr = 1e-2
            args.patience = 50
        if args.dataset == "plym-density_prop":
            # polymer density
            args.epochs = 400
            args.l2reg = 1e-5
            args.gamma = 0.3
            args.num_layer = 3
            args.drop_ratio = 0.5
            if args.gnn == "gcn-virtual":
                args.l2reg = 1e-4
            args.batch_size = 32
            args.lr = 1e-3
            args.patience = 50
            args.use_clip_norm = True

        if args.dataset == "ogbg-molhiv":
            args.gamma = 0.1
            args.batch_size = 512
            args.initw_name = "orthogonal"
            if args.gnn == "gcn-virtual":
                args.lr = 1e-3
                args.l2reg = 1e-5
                args.epochs = 100
                args.num_layer = 3
                args.use_clip_norm = True
                args.path_list = [2, 4]
        if args.dataset == "ogbg-molbace":
            if args.gnn == "gin-virtual" or args.gnn == "gin":
                args.gnn = "gin"
                args.l2reg = 7e-4
                args.gamma = 0.55
                args.num_layer = 4
                args.batch_size = 64
                args.emb_dim = 64
                args.use_lr_scheduler = True
                args.patience = 100
                args.drop_ratio = 0.3
                args.initw_name = "orthogonal"
            if args.gnn == "gcn-virtual" or args.gnn == "gcn":
                args.gnn = "gcn"
                args.patience = 100
                args.initw_name = "orthogonal"
                args.num_layer = 2
                args.emb_dim = 64
                args.batch_size = 128
        if args.dataset == "ogbg-molbbbp":
            args.l2reg = 5e-6
            args.initw_name = "orthogonal"
            args.num_layer = 2
            args.emb_dim = 64
            args.batch_size = 256
            args.use_lr_scheduler = True
            args.gamma = 0.2
            if args.gnn == "gcn-virtual" or args.gnn == "gcn":
                args.gnn = "gcn-virtual"
                args.gamma = 0.4
                args.emb_dim = 128
                args.use_lr_scheduler = False
        if args.dataset == "ogbg-molsider":
            if args.gnn == "gin-virtual" or args.gnn == "gin":
                args.gnn = "gin"
            if args.gnn == "gcn-virtual" or args.gnn == "gcn":
                args.gnn = "gcn"
            args.l2reg = 1e-4
            args.patience = 100
            args.gamma = 0.65
            args.num_layer = 5
            args.epochs = 400
        if args.dataset == "ogbg-molclintox":
            if args.gnn == "gin-virtual" or args.gnn == "gin":
                args.gnn = "gin"
            if args.gnn == "gcn-virtual" or args.gnn == "gcn":
                args.gnn = "gcn"
            args.use_linear_predictor = True
            args.use_clip_norm = True
            args.gamma = 0.2
            args.patience = 100
            args.batch_size = 64
            args.num_layer = 5
            args.emb_dim = 300
            args.l2reg = 1e-4
            args.epochs = 400
            args.drop_ratio = 0.5
        if args.dataset == "ogbg-moltox21":
            args.gamma = 0.8
        if args.dataset == "ogbg-moltoxcast":
            if args.gnn == "gin-virtual" or args.gnn == "gin":
                args.gnn = "gin"
            if args.gnn == "gcn-virtual" or args.gnn == "gcn":
                args.gnn = "gcn"
            args.patience = 50
            args.epochs = 150
            args.l2reg = 1e-5
            args.gamma = 0.7
            args.num_layer = 2

    # args.plym_prop = 'none' if args.dataset.startswith('ogbg') else args.dataset.split('-')[1].split('_')[0]
    args.plym_prop = TARGET
    if args.dataset.startswith("ogbg"):
        results = {"valid_auc": [], "test_auc": []}
    else:
        results = {"valid_rmse": [], "test_rmse": [], "test_r2": []}
    for _ in range(args.trails):
        if args.dataset.startswith("plym"):
            valid_rmse, test_rmse, test_r2 = main(args)
            results["test_r2"].append(test_r2)
            results["test_rmse"].append(test_rmse)
            results["valid_rmse"].append(valid_rmse)
        else:
            valid_auc, test_auc = main(args)
            results["valid_auc"].append(valid_auc)
            results["test_auc"].append(test_auc)
    for mode, nums in results.items():
        print("{}: {:.4f}+-{:.4f} {}".format(mode, np.mean(nums), np.std(nums), nums))


def graph_decode_test(smiles):
    g = smiles2graph(smiles)
    s = graph2smiles(g)
    if smiles != s:
        print(f"{smiles} => {s}")

if __name__ == "__main__":
    
    print(smiles2graph("C1=CC=CC=C1"))
    # df_full = pd.read_csv(DATA_DIR + "/" + TRAIN_FILE_NAME, engine="python")
    # for smiles in df_full["SMILES"].tolist():
    #     graph_decode_test(smiles)

    # exit(0)
    args = get_args()
    config_and_run(args)
