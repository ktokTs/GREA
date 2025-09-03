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

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

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


class PolymerRegDataset(InMemoryDataset):
    def __init__(self, name="o2_prop", root="data", transform=None, pre_transform=None):
        """
        - name (str): name of the dataset
        - root (str): root directory to store the dataset folder
        - transform, pre_transform (optional): transform/pre-transform graph objects
        """
        self.name = name
        self.dir_name = "_".join(name.split("-"))
        root = osp.join(root, name, "raw")
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
        read_path = osp.join(self.original_root, self.name.split("_")[0] + "_raw.csv")
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
        graph_list = []
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


nn_act = torch.nn.ReLU()  # ReLU()
F_act = F.relu


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(GINConv, self).__init__(aggr="add")

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
        external_only=True
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
        h_node = self.graph_encoder(batched_data)
        h_r, h_env, r_node_num, env_node_num = self.separator(batched_data, h_node)
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
    
    def encode(self, batched_data):
        """
        推論用特徴抽出。
        external_only=True の場合は rationale 部分 (h_r)、
        False の場合は rationale を返す（用途未定なら h_r で十分）。
        """
        with torch.no_grad():
            h_node = self.graph_encoder(batched_data)
            h_r, h_env, _, _ = self.separator(batched_data, h_node)
            if self.external_only:
                return h_r
            else:
                return h_r  # 拡張予定: h_r と h_env の結合など


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


try:
    import xgboost as xgb
except ImportError:
    raise ImportError("xgboost 未インストールです: pip install xgboost")


def get_args():
    p = argparse.ArgumentParser(description="GNN encoder + XGBoost")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--dataset", type=str, default="ogbg-molbbbp")
    p.add_argument("--gnn", type=str, default="gin-virtual")
    p.add_argument("--num_layer", type=int, default=5)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--drop_ratio", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--init_seed", type=int, default=0)
    p.add_argument("--trails", type=int, default=1)
    # XGBoost ハイパーパラメータ
    p.add_argument("--xgb_estimators", type=int, default=500)
    p.add_argument("--xgb_lr", type=float, default=0.05)
    p.add_argument("--xgb_max_depth", type=int, default=6)
    p.add_argument("--early_stop", type=int, default=30)
    return p.parse_args()


def collect_reps(args, model, loader, device, is_regression, plym_prop=None):
    feats = []
    ys = []
    for batch in loader:
        batch = batch.to(device)
        if batch.x.size(0) == 1:
            continue
        with torch.no_grad():
            h = model.encode(batch)  # (num_graphs, emb_dim)
        if is_regression:
            if plym_prop == "density":
                y = torch.log(batch[plym_prop])
            elif plym_prop and plym_prop != "none":
                y = batch[plym_prop]
            else:
                y = batch.y
        else:
            y = batch.y
        feats.append(h.cpu())
        ys.append(y.view(h.size(0), -1).cpu())
    if not feats:
        return None, None
    X = torch.cat(feats, 0).numpy()
    Y = torch.cat(ys, 0).numpy()
    return X, Y


def run_once(args, seed_offset=0):
    torch.manual_seed(args.init_seed + seed_offset)
    np.random.seed(args.init_seed + seed_offset)

    device = (
        torch.device(f"cuda:{args.device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if args.dataset.startswith("ogbg"):
        dataset = PygGraphPropPredDataset(name=args.dataset, root="data")
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(
            dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=False
        )
        valid_loader = DataLoader(
            dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False
        )
        evaluator = Evaluator(args.dataset)
        is_classification = "classification" in dataset.task_type
        plym_prop = None
        num_tasks = dataset.num_tasks
    else:
        # Polymer regression
        dataset = PolymerRegDataset(name=args.dataset.split("-")[1], root="data")
        full_idx = list(range(len(dataset)))
        train_ratio, valid_ratio, test_ratio = 0.6, 0.1, 0.3
        train_index, test_index, _, _ = train_test_split(
            full_idx, full_idx, test_size=test_ratio, random_state=42
        )
        train_index, val_index, _, _ = train_test_split(
            train_index,
            train_index,
            test_size=valid_ratio / (valid_ratio + train_ratio),
            random_state=42,
        )
        train_loader = DataLoader(
            dataset[torch.LongTensor(train_index)],
            batch_size=args.batch_size,
            shuffle=False,
        )
        valid_loader = DataLoader(
            dataset[torch.LongTensor(val_index)],
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            dataset[torch.LongTensor(test_index)],
            batch_size=args.batch_size,
            shuffle=False,
        )
        evaluator = Evaluator("ogbg-molesol")  # RMSE evaluator
        is_classification = False
        plym_prop = args.dataset.split("-")[1].split("_")[0]
        num_tasks = 1

    model = GraphEnvAug(
        num_tasks=num_tasks,
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        gnn_type=args.gnn,
        drop_ratio=args.drop_ratio,
        gamma=args.gamma,
    ).to(device)

    # (簡略化) 重み初期化はデフォルト (必要なら独自 init を追加)

    X_train, y_train = collect_reps(
        args, model, train_loader, device, not is_classification, plym_prop
    )
    X_valid, y_valid = collect_reps(
        args, model, valid_loader, device, not is_classification, plym_prop
    )
    X_test, y_test = collect_reps(
        args, model, test_loader, device, not is_classification, plym_prop
    )

    if X_train is None:
        raise RuntimeError("特徴量が空です。")

    if is_classification:
        # タスク毎に独立モデル (欠損ラベル NaN 対応)
        preds_valid = []
        preds_test = []
        for t in range(y_train.shape[1]):
            mask_tr = ~np.isnan(y_train[:, t])
            mask_va = ~np.isnan(y_valid[:, t])
            dtrain = xgb.DMatrix(X_train[mask_tr], label=y_train[mask_tr, t])
            dvalid = xgb.DMatrix(X_valid[mask_va], label=y_valid[mask_va, t])
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "eta": args.xgb_lr,
                "max_depth": args.xgb_max_depth,
            }
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=args.xgb_estimators,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=args.early_stop,
                verbose_eval=False,
            )
            preds_valid.append(booster.predict(xgb.DMatrix(X_valid)))
            preds_test.append(booster.predict(xgb.DMatrix(X_test)))
        y_pred_valid = np.vstack(preds_valid).T
        y_pred_test = np.vstack(preds_test).T
        input_valid = {"y_true": y_valid, "y_pred": y_pred_valid}
        input_test = {"y_true": y_test, "y_pred": y_pred_test}
        valid_auc = evaluator.eval(input_valid)["rocauc"]
        test_auc = evaluator.eval(input_test)["rocauc"]
        print({"Valid AUC": valid_auc, "Test AUC": test_auc})
        return valid_auc, test_auc
    else:
        # 単一タスク回帰
        from sklearn.metrics import mean_squared_error, r2_score

        reg = xgb.XGBRegressor(
            n_estimators=args.xgb_estimators,
            learning_rate=args.xgb_lr,
            max_depth=args.xgb_max_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            early_stopping_rounds=args.early_stop,
        )
        reg.fit(
            X_train,
            y_train.reshape(-1),
            eval_set=[(X_valid, y_valid.reshape(-1))],
            verbose=False,
        )
        pred_valid = reg.predict(X_valid)
        pred_test = reg.predict(X_test)
        rmse_valid = mean_squared_error(y_valid, pred_valid, squared=False)
        rmse_test = mean_squared_error(y_test, pred_test, squared=False)
        r2 = r2_score(y_test, pred_test)
        print({"Valid RMSE": rmse_valid, "Test RMSE": rmse_test, "Test R2": r2})
        return rmse_valid, rmse_test, r2


def main():
    args = get_args()
    if args.dataset.startswith("ogbg"):
        results_valid = []
        results_test = []
        for i in range(args.trails):
            v, t = run_once(args, seed_offset=i)
            results_valid.append(v)
            results_test.append(t)
        print(
            "Valid AUC avg: {:.4f} +/- {:.4f}".format(
                np.mean(results_valid), np.std(results_valid)
            )
        )
        print(
            "Test  AUC avg: {:.4f} +/- {:.4f}".format(
                np.mean(results_test), np.std(results_test)
            )
        )
    else:
        val_rmse_list, test_rmse_list, test_r2_list = [], [], []
        for i in range(args.trails):
            v_rmse, t_rmse, t_r2 = run_once(args, seed_offset=i)
            val_rmse_list.append(v_rmse)
            test_rmse_list.append(t_rmse)
            test_r2_list.append(t_r2)
        print(
            "Valid RMSE avg: {:.4f} +/- {:.4f}".format(
                np.mean(val_rmse_list), np.std(val_rmse_list)
            )
        )
        print(
            "Test  RMSE avg: {:.4f} +/- {:.4f}".format(
                np.mean(test_rmse_list), np.std(test_rmse_list)
            )
        )
        print(
            "Test  R2   avg: {:.4f} +/- {:.4f}".format(
                np.mean(test_r2_list), np.std(test_r2_list)
            )
        )


if __name__ == "__main__":
    main()
