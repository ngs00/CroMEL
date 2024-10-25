import numpy
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def calc_transferability(model, dataset_x, dataset_y, alpha=1.0):
    with torch.no_grad():
        z = model.generator(dataset_x.cuda()).cpu().numpy()
    lin_model = Ridge(alpha=alpha)
    lin_model.fit(z, dataset_y.numpy())
    train_loss = numpy.sqrt(mean_squared_error(dataset_y.numpy(), lin_model.predict(z)))

    return 1 / train_loss
