import torch.nn as nn
import torch


class Linear(nn.Module):
    def __init__(self, input_size, batch_size, output_size):
        super(Linear, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(0.5)
        # self.linear2 = nn.Linear(input_size // 2, output_size)
        # self.linear3 = nn.Linear(output_size * 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        # x = self.dropout(x)
        # x = self.linear2(x)
        # x = self.linear3(x)
        return self.relu(x)
