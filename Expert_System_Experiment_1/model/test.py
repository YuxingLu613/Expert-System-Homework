import torch


def test(model, test_data):
    model.load_state_dict(torch.load("./best_model.pth"))
    model.eval()
    return model(test_data)
