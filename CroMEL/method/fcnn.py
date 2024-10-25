import torch
from torch.nn.functional import leaky_relu


class FCNN(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(FCNN, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hidden)
        self.fc2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = torch.nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        h = leaky_relu(self.fc1(x))
        h = leaky_relu(self.fc2(h))
        out = self.fc3(h)

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
            return torch.vstack([self(x.cuda()) for x, _ in data_loader]).cpu().numpy()
