import numpy
import pandas
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from method.gnn import *
from method.model import *
from util.chem import load_elem_attrs
from util.data_exp import load_dataset, get_k_fold


dim_hidden = 64
src_dataset = 'mp'
src_targets = ['fe', 'bg']
emb_models = ['mpnn', 'cgcnn']


random_seed = 0
dataset_name = 'efe'
idx_target = 3
idx_composition = 0
num_folds = 5
batch_size = 32
init_lr = 5e-4
l2_reg_coeff = 5e-6
num_epochs = 1000
elem_attrs = load_elem_attrs('res/matscholar-embedding.json')


dataset = load_dataset('dataset/{}.xlsx'.format(dataset_name), elem_attrs, idx_composition, idx_target, False)
dict_r2 = dict()
dict_mae = dict()


for src_target in src_targets:
    for emb_model in emb_models:
        if emb_model == 'mpnn':
            emb_net = MPNN(dim_node_feat=200,
                           dim_edge_feat=dim_hidden,
                           dim_hidden=dim_hidden,
                           dim_out=dim_hidden)
        elif emb_model == 'cgcnn':
            emb_net = CGCNN(dim_node_feat=200,
                            dim_edge_feat=dim_hidden,
                            dim_hidden=dim_hidden,
                            dim_out=dim_hidden)
        else:
            raise KeyError

        generator = Generator(dim_composition_vec=600, dim_out=dim_hidden)
        model_src = CrossModNet(emb_net=emb_net,
                                generator=generator,
                                dim_hidden=dim_hidden,
                                dim_out=1).cuda()
        model_src.load_state_dict(torch.load('res/{}_{}_{}_cromel.pt'.format(src_dataset, src_target, emb_model)))
        id_src_model = '{}_{}_{}'.format(src_dataset, src_target, emb_model)
        dict_r2[id_src_model] = list()
        dict_mae[id_src_model] = list()

        print('------------------------- {} {} -------------------------'.format(dataset_name, id_src_model))
        for k in range(0, num_folds):
            dataset_train, dataset_test = get_k_fold(dataset, num_folds, idx_fold=k, random_seed=random_seed)
            target_test = dataset_test[:, -1].view(-1, 1).numpy()
            dataset_train = TensorDataset(dataset_train[:, :-1], dataset_train[:, -1].view(-1, 1))
            dataset_test = TensorDataset(dataset_test[:, :-1], dataset_test[:, -1].view(-1, 1))
            loader_train = DataLoader(dataset_train, batch_size, shuffle=True)
            loader_test = DataLoader(dataset_test, batch_size)

            model = FCNN_TL(model_src.generator, dim_hidden=dim_hidden, dim_out=1).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=l2_reg_coeff)
            loss_func = torch.nn.L1Loss()

            for epoch in range(0, num_epochs):
                loss_train = model.fit(loader_train, optimizer, loss_func)
                print('Fold [{}/{}]\tEpoch [{}/{}]\tTraining loss: {:.3f}'
                      .format(k + 1, num_folds, epoch + 1, num_epochs, loss_train))

            preds_test = model.predict(loader_test)
            dict_r2[id_src_model].append(r2_score(target_test, preds_test))
            dict_mae[id_src_model].append(mean_absolute_error(target_test, preds_test))

            pred_results = pandas.DataFrame(numpy.hstack([target_test, preds_test]))
            pred_results.to_excel('save/preds_{}_{}_{}_{}.xlsx'.format(dataset_name, idx_target, id_src_model, k),
                                  index=False, header=False)
            torch.save(model.state_dict(), 'save/model_{}_{}_{}_{}.pt'.format(dataset_name, idx_target, id_src_model, k))


print('========================= Transfer learning results =========================')
for id_src_model in dict_r2.keys():
    print('Souce model: ' + id_src_model)
    print('R2-score: {:.3f} ({:.3f})'.format(numpy.mean(dict_r2[id_src_model]), numpy.std(dict_r2[id_src_model])))
    print('MAE: {:.3f} ({:.3f})'.format(numpy.mean(dict_mae[id_src_model]), numpy.std(dict_mae[id_src_model])))
    print('-----------------------------------------------')
