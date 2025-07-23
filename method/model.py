import torch
import itertools
from torch.nn.functional import normalize, leaky_relu
from method.gnn import MPNN, CGCNN


class Generator(torch.nn.Module):
    def __init__(self, dim_composition_vec, dim_out):
        super(Generator, self).__init__()
        self.emb_net_cv = torch.nn.Sequential(
            torch.nn.Linear(dim_composition_vec, dim_out),
            torch.nn.PReLU(),
            torch.nn.Linear(dim_out, dim_out),
            torch.nn.PReLU()
        )
        self.fc_std = torch.nn.Sequential(
            torch.nn.Linear(dim_out, dim_out),
            torch.nn.Softplus()
        )

    def forward(self, composition_vec):
        hc = self.emb_net_cv(composition_vec)
        hc_std = self.fc_std(hc)

        return hc + torch.randn_like(hc_std) * hc_std


class CrossModNet(torch.nn.Module):
    def __init__(self, emb_net, generator, dim_hidden, dim_out):
        super(CrossModNet, self).__init__()
        self.num_critic_opt = 5
        self.emb_net = emb_net
        self.generator = generator
        self.pred_net = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden, dim_out),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden, dim_hidden),
            torch.nn.PReLU(),
            torch.nn.Linear(dim_hidden, 1)
        )

        self.pred_params = itertools.chain(self.emb_net.parameters(), self.pred_net.parameters())
        self.opt_critic = torch.optim.RMSprop(self.critic.parameters(), lr=1e-4)
        self.opt_generator = torch.optim.RMSprop(self.generator.parameters(), lr=1e-4)

    def emb(self, x, edge_index, edge_attr, batch):
        return normalize(self.emb_net(x, edge_index, edge_attr, batch), p=2, dim=1)

    def forward(self, x, edge_index, edge_attr, batch):
        embs = self.emb(x, edge_index, edge_attr, batch)
        out = self.pred_net(embs)

        return out

    def fit(self, data_loader, optimizer):
        num_batches = len(data_loader)
        loss_train = 0
        loss_critic = 0
        loss_generator = 0

        self.train()
        for n, batch in enumerate(data_loader):
            batch = batch.cuda()

            # Train the critic model.
            emb_hs = self.emb(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            emb_hc = self.generator(batch.composition_vec)
            critic_hs = self.critic(emb_hs)
            critic_hc = self.critic(emb_hc)
            grad = torch.autograd.grad(torch.mean(critic_hc), emb_hc, create_graph=True,
                                       retain_graph=True, only_inputs=True)[0]
            gp = torch.mean((torch.linalg.norm(grad, dim=1) - 1)**2)

            loss_crt = -torch.mean(critic_hs) + torch.mean(critic_hc) + 10 * gp
            self.opt_critic.zero_grad()
            loss_crt.backward()
            self.opt_critic.step()
            loss_critic += loss_crt.detach().item()

            # Train the generator model.
            if (n + 1) % self.num_critic_opt == 0:
                critic_hc = self.critic(self.generator(batch.composition_vec))
                loss_gen = -torch.mean(critic_hc)
                self.opt_generator.zero_grad()
                loss_gen.backward()
                self.opt_generator.step()
                loss_generator += loss_gen.detach().item()

            # Train the embedding and prediction networks.
            preds = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss_pred = torch.sum((batch.y - preds)**2)
            optimizer.zero_grad()
            loss_pred.backward()
            optimizer.step()
            loss_train += loss_pred.detach().item()

        loss_train /= num_batches
        loss_critic /= (self.num_critic_opt * num_batches)
        loss_generator /= num_batches

        return loss_train, loss_critic, loss_generator

    def predict(self, data_loader):
        list_preds = list()

        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.cuda()
                list_preds.append(self(batch.x, batch.edge_index, batch.edge_attr, batch.batch))

        return torch.vstack(list_preds).cpu().numpy()


class FCNN_TL(torch.nn.Module):
    def __init__(self, generator, dim_hidden, dim_out):
        super(FCNN_TL, self).__init__()
        self.generator = generator
        self.fc1 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc2 = torch.nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        h = self.generator(x)
        h = leaky_relu(self.fc1(h))
        out = self.fc2(h)

        return out

    def fit(self, data_loader, optimizer, loss_func):
        loss_train = 0

        self.train()
        for x, y in data_loader:
            preds = self(x.cuda())
            loss = loss_func(preds, y.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.detach().item()

        return loss_train / len(data_loader)

    def predict(self, data_loader):
        self.eval()
        with torch.no_grad():
            return torch.vstack([self(x.cuda()) for x, _ in data_loader]).cpu()


def get_emb_model(name_gnn, dim_node_feat, dim_edge_feat, dim_hidden):
    if name_gnn == 'mpnn':
        return MPNN(dim_node_feat=dim_node_feat,
                    dim_edge_feat=dim_edge_feat,
                    dim_hidden=dim_hidden,
                    dim_out=dim_hidden).cuda()
    elif name_gnn == 'cgcnn':
        return CGCNN(dim_node_feat=dim_node_feat,
                     dim_edge_feat=dim_edge_feat,
                     dim_hidden=dim_hidden,
                     dim_out=dim_hidden)
    else:
        raise KeyError
