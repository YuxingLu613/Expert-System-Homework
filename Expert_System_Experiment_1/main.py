from Expert_System_Expirement_1.data.process import get_data, get_label, get_test_data, get_test_label
from Expert_System_Expirement_1.data.prepare import load
from Expert_System_Expirement_1.model.module import Linear
from Expert_System_Expirement_1.model.train import train
from Expert_System_Expirement_1.model.test import test
from Expert_System_Expirement_1.config import Config
from Expert_System_Expirement_1 import plot

import numpy as np
import pandas

import torch

if __name__ == "__main__":
    config = Config()
    raw_train_data = get_data(config.train_data_path)
    train_data, train_label = get_label(raw_train_data)
    input_size = np.shape(train_data)[1]
    raw_test_data = get_test_data(config.test_data_path)
    test_data = get_test_label(raw_test_data)

    train_loader = load(train_data, train_label)

    if not config.device:
        model = Linear(input_size, config.batch_size, 1)
    else:
        model = Linear(input_size, config.batch_size, 1).cuda()
    loss_fn = torch.nn.MSELoss()
    learning_rate = config.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)

    loss_lst = train(train_loader, model, loss_fn, optimizer)
    plot.plot_loss_curve(loss_lst)
    test_result = test(model, test_data)
    print(test_result)
    data = pandas.DataFrame(test_result.detach().numpy())
    data.columns = ["result"]
    data.to_csv(config.output_path)
