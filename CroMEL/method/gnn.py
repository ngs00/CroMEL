import torch
from torch.nn.functional import normalize, leaky_relu
from torch_geometric.nn.conv import NNConv, CGConv, GINConv
from torch_geometric.nn.glob import global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_hidden, dim_out):
        super(GNN, self).__init__()
        self.fc1 = torch.nn.Linear(dim_node_feat, dim_hidden)
        self.fc2 = torch.nn.Linear(dim_hidden, dim_out)
        self.act_fc1 = torch.nn.PReLU()
        self.act_gc1 = torch.nn.PReLU()
        self.act_gc2 = torch.nn.PReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.act_fc1(self.fc1(x))
        h = self.act_gc1(self.gc1(h, edge_index, edge_attr))
        h = self.act_gc2(self.gc2(h, edge_index, edge_attr))
        hg = normalize(global_mean_pool(h, batch), p=2, dim=1)
        out = self.fc2(hg)

        return out

    def fit(self, data_loader, optimizer):
        loss_train = 0

        self.train()
        for batch in data_loader:
            batch = batch.cuda()
            preds = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = torch.sum((batch.y - preds)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.detach().item()

        return loss_train / len(data_loader)

    def predict(self, data_loader):
        list_preds = list()

        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.cuda()
                list_preds.append(self(batch.x, batch.edge_index, batch.edge_attr, batch.batch))

        return torch.vstack(list_preds).cpu().numpy()


class GIN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_hidden, dim_out):
        super(GIN, self).__init__()
        self.gc1 = GINConv(torch.nn.Linear(dim_node_feat, dim_hidden))
        self.gc2 = GINConv(torch.nn.Linear(dim_hidden, dim_hidden))
        self.fc = torch.nn.Linear(dim_hidden, dim_out)

    def forward(self, x, edge_index, batch):
        h = leaky_relu(self.gc1(x, edge_index))
        h = leaky_relu(self.gc2(h, edge_index))
        hg = normalize(global_mean_pool(h, batch), p=2, dim=1)
        out = self.fc(hg)

        return out

    def predict(self, data_loader):
        list_preds = list()

        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.cuda()
                list_preds.append(self(batch.x, batch.edge_index, batch.batch))

        return torch.vstack(list_preds).cpu().numpy()


class MPNN(GNN):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_hidden, dim_out):
        super(MPNN, self).__init__(dim_node_feat, dim_hidden, dim_out)
        self.efc1 = torch.nn.Sequential(torch.nn.Linear(dim_edge_feat, dim_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(dim_hidden, dim_hidden * dim_hidden))
        self.gc1 = NNConv(dim_hidden, dim_hidden, self.efc1)
        self.efc2 = torch.nn.Sequential(torch.nn.Linear(dim_edge_feat, dim_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(dim_hidden, dim_hidden * dim_hidden))
        self.gc2 = NNConv(dim_hidden, dim_hidden, self.efc2)


class CGCNN(GNN):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_hidden, dim_out):
        super(CGCNN, self).__init__(dim_node_feat, dim_hidden, dim_out)
        self.gc1 = CGConv(dim_hidden, dim_edge_feat)
        self.gc2 = CGConv(dim_hidden, dim_edge_feat)
