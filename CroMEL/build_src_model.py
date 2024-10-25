import numpy
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from util.chem import load_elem_attrs
from util.data import load_dataset
from method.gnn import *
from method.model import CrossModNet, Generator


random_seed = 0
dim_hidden = 64
init_lr = 5e-4
l2_reg_coeff = 1e-6
batch_size = 64
num_epochs = 300
name_dataset = 'mp'
name_targets = ['fe', 'bg']
idx_target_by_name = {
    'nlhm': {'bg_gga': 1, 'bg': 2},
    'cmr_abs3': {'bg': 2},
    'cmr_abse3': {'hf': 2},
    'cmr_c1db': {'bg': 2},
    'cmr_c2db': {'hf': 2, 'ehull': 3, 'bg': 4, 'bg_hse': 5, 'bg_gw': 6},
    'qmof': {'magmom_pbe': 1, 'magmom_hse': 2, 'bg_pbe': 3, 'bg_hse': 4},
    'mps': {'sm': 4, 'bm': 5},
    'mp': {'fe': 4, 'bg': 5}
}
name_emb_model = 'cgcnn'

for name_target in name_targets:
    idx_target = idx_target_by_name[name_dataset][name_target]
    elem_attrs = load_elem_attrs('res/matscholar-embedding.json')
    dataset = load_dataset(path_metadata='source_dataset/{}/metadata.xlsx'.format(name_dataset),
                           path_structs='source_dataset/{}/struct'.format(name_dataset),
                           elem_attrs=elem_attrs,
                           idx_struct=0,
                           idx_target=idx_target)
    loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader_eval = DataLoader(dataset, batch_size=128, shuffle=False)
    print(name_dataset, name_target, len(dataset))

    if name_emb_model == 'mpnn':
        emb_net = MPNN(dim_node_feat=dataset[0].x.shape[1],
                       dim_edge_feat=dataset[0].edge_attr.shape[1],
                       dim_hidden=dim_hidden,
                       dim_out=dim_hidden).cuda()
    elif name_emb_model == 'cgcnn':
        emb_net = CGCNN(dim_node_feat=dataset[0].x.shape[1],
                        dim_edge_feat=dataset[0].edge_attr.shape[1],
                        dim_hidden=dim_hidden,
                        dim_out=dim_hidden)
    else:
        raise KeyError

    generator = Generator(dim_composition_vec=dataset[0].composition_vec.shape[1], dim_out=dim_hidden)
    model = CrossModNet(emb_net=emb_net,
                        generator=generator,
                        dim_hidden=dim_hidden,
                        dim_out=1).cuda()
    optimizer = torch.optim.Adam(model.pred_params, lr=init_lr, weight_decay=l2_reg_coeff)

    for epoch in range(0, num_epochs):
        loss_train, loss_critic, loss_generator = model.fit(loader_train, optimizer)

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), 'save/{}_{}_{}_cromel.pt'.format(name_dataset, name_target, name_emb_model))
            y_val = numpy.vstack([d.y.item() for d in dataset])
            y_pred = model.predict(loader_eval)
            print('R2 score: {:.3f}'.format(r2_score(y_val, y_pred)))

        print('Epoch [{}/{}]\tTraining loss: {:.3f}\tCritic loss: {:.3f}\tGenerator loss: {:.3f}'
              .format(epoch + 1, num_epochs, loss_train, loss_critic, loss_generator))

    torch.save(model.state_dict(), 'save/{}_{}_{}_cromel.pt'.format(name_dataset, name_target, name_emb_model))
