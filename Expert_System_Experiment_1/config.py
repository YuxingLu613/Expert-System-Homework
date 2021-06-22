import torch


class Config():
    # device = True if torch.cuda.is_available() else False
    device = False
    train_data_path = "./data/train.csv"
    test_data_path = "./data/test.csv"
    output_path = "result.csv"

    batch_size = 64
    epoches = 20000
    lr = 0.00007
