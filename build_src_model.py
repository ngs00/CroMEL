import numpy
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from util.chem import load_elem_attrs
from util.data import load_calc_dataset
from method.model import CrossModNet, Generator, get_emb_model


dim_hidden = 64
num_epochs = 300
name_dataset = 'mps'
name_target = 'fe'
name_emb_model = 'cgcnn'


# Load a source calculation dataset for building a source model.
dataset = load_calc_dataset(path_metadata='dataset/src_calc/{}/metadata.xlsx'.format(name_dataset),
                            path_structs='dataset/src_calc/{}/struct'.format(name_dataset),
                            elem_attrs=load_elem_attrs('res/matscholar-embedding.json'),
                            idx_struct=0,
                            idx_target=2)
loader_train = DataLoader(dataset, batch_size=64, shuffle=True)
loader_eval = DataLoader(dataset, batch_size=128, shuffle=False)
y_val = numpy.vstack([d.y.item() for d in dataset])


# Define a structure embedding network and a CroMEL model.
emb_net = get_emb_model(name_gnn=name_emb_model,
                        dim_node_feat=dataset[0].x.shape[1],
                        dim_edge_feat=dataset[0].edge_attr.shape[1],
                        dim_hidden=dim_hidden)
generator = Generator(dim_composition_vec=dataset[0].composition_vec.shape[1], dim_out=dim_hidden)
model = CrossModNet(emb_net=emb_net, generator=generator, dim_hidden=dim_hidden, dim_out=1).cuda()


# Optimize model parameters on the source calculation dataset.
optimizer = torch.optim.Adam(model.pred_params, lr=5e-4, weight_decay=1e-6)
for epoch in range(0, num_epochs):
    loss_train, loss_critic, loss_generator = model.fit(loader_train, optimizer)
    print('Epoch [{}/{}]\tTraining loss: {:.3f}\tCritic loss: {:.3f}\tGenerator loss: {:.3f}'
          .format(epoch + 1, num_epochs, loss_train, loss_critic, loss_generator))

    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), 'save/{}_{}_{}_cromel.pt'.format(name_dataset, name_target, name_emb_model))
        y_pred = model.predict(loader_eval)
        print('R2 score: {:.3f}'.format(r2_score(y_val, y_pred)))
torch.save(model.state_dict(), 'save/{}_{}_{}_cromel.pt'.format(name_dataset, name_target, name_emb_model))
