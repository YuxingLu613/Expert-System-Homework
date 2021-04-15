from torch.utils.data import DataLoader, TensorDataset
from ..config import Config


def load(x, y):
    train_tensorset = TensorDataset(x, y)
    train_loader = DataLoader(dataset=train_tensorset, batch_size=Config.batch_size, shuffle=True)
    return train_loader
