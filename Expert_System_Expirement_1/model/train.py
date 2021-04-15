from tqdm import tqdm
from ..config import Config
import torch


def train(train_loader, model, loss_fn, optimizer):
    loss_lst = []
    best_loss = 999999
    with tqdm(total=Config.epoches) as t:
        for epoch in range(Config.epoches):
            total_loss = 0
            for x, y in enumerate(train_loader):
                y1 = model(y[0])
                loss = loss_fn(y1, y[1])
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            t.set_description("epoch %d" % epoch)
            t.set_postfix(loss="%d" % total_loss)
            if epoch % 100 == 0:
                loss_lst.append(total_loss)
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), "./data/best_model.pth")
            t.update(1)
    return loss_lst
