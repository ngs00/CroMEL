import numpy
import pandas
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from util.chem import load_elem_attrs
from util.data import load_exp_dataset
from method.model import *


# Experiment settings.
random_seed = 0
num_folds = 5
num_epochs = 1000
idx_tgt_target = 3


# Load a source model.
name_src_dataset = 'mp'
name_src_target = 'fe'
name_emb_model = 'cgcnn'
dim_node_feat = 200
dim_edge_feat = 64
dim_hidden = 64
emb_net = get_emb_model(name_gnn=name_emb_model,
                        dim_node_feat=dim_node_feat,
                        dim_edge_feat=dim_edge_feat,
                        dim_hidden=dim_hidden)
generator = Generator(dim_composition_vec=600, dim_out=dim_hidden)
model_src = CrossModNet(emb_net=emb_net,
                        generator=generator,
                        dim_hidden=dim_hidden,
                        dim_out=1).cuda()
model_src.load_state_dict(torch.load('save/{}_{}_{}_cromel.pt'
                                     .format(name_src_dataset, name_src_target, name_emb_model)))


# Load the target experimental dataset.
name_tgt_dataset = 'efe'
exp_dataset = load_exp_dataset(path_dataset='dataset/tgt_exp/{}.xlsx'.format(name_tgt_dataset),
                               elem_attrs=load_elem_attrs('res/matscholar-embedding.json'),
                               idx_composition=0,
                               idx_target=idx_tgt_target,
                               log_target=False)
k_folds = exp_dataset.get_k_folds(num_folds, random_seed)


# Train and evaluate based on k-fold cross validation.
list_r2 = list()
list_mae = list()
for k in range(0, num_folds):
    dataset_train = k_folds[k][0]
    dataset_test = k_folds[k][1]
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=128)
    model = FCNN_TL(model_src.generator, dim_hidden=dim_hidden, dim_out=1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-6)
    loss_func = torch.nn.L1Loss()

    # Optimize model parameters of the target prediction model.
    for epoch in range(0, num_epochs):
        loss_train = model.fit(loader_train, optimizer, loss_func)

        if (epoch + 1) % 100 == 0:
            print('Fold [{}/{}]\tEpoch [{}/{}]\tTraining loss: {:.3f}'
                  .format(k + 1, num_folds, epoch + 1, num_epochs, loss_train))

    # Evaluate the trained model on the test dataset.
    targets_test = dataset_test.y.numpy()
    preds_test = model.predict(loader_test).numpy()
    list_r2.append(r2_score(targets_test, preds_test))
    list_mae.append(mean_absolute_error(targets_test, preds_test))

    pred_results = pandas.DataFrame(numpy.hstack([targets_test, preds_test]))
    pred_results.to_excel('save/preds_{}_{}_{}_.xlsx'.format(name_tgt_dataset, idx_tgt_target, k),
                          index=False, header=False)

print('------------------- Evaluation Results -------------------')
print('R2-score: {:.3f} (\u00B1{:.3f})'.format(numpy.mean(list_r2), numpy.std(list_r2)))
print('MAE: {:.3f} (\u00B1{:.3f})'.format(numpy.mean(list_mae), numpy.std(list_mae)))
