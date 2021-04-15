import matplotlib.pyplot as plt
from .config import Config


def plot_loss_curve(loss_lst):
    index = [i for i in range(Config.epoches // 100)]
    plt.plot(index, loss_lst)
    plt.show()
    plt.savefig("./loss.png")
