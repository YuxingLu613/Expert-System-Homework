import pandas as pd
import numpy as np
import torch
from ..config import Config


def get_data(path):
    data = pd.read_csv(path, encoding="big5")
    data = data.iloc[:, 3:]
    data[data == "NR"] = 0
    raw_data = data.to_numpy()
    return raw_data


def normalization(data, _range):
    return (data - np.min(data)) / _range


def get_label(raw_data):
    month_data = {}
    for month in range(12):
        sample = np.empty([18, 480])
        for day in range(20):
            sample[:, day * 24:(day + 1) * 24] = raw_data[18 * (20 * month + day):18 * (20 * month + day + 1), :]
        month_data[month] = sample

    x = np.empty([12 * 471, 18 * 9], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour:day * 24 + hour + 9].reshape(
                    1, -1)
                y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]
    global rangex
    rangex = np.max(x) - np.min(x)
    if Config.device:
        x = torch.tensor(normalization(x, rangex), dtype=torch.float32).cuda()
        y = torch.tensor(y, dtype=torch.float32).cuda()
    else:
        x = torch.tensor(normalization(x, rangex), dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
    return x, y


def get_test_data(path):
    data = pd.read_csv(path, encoding="big5")
    data = data.iloc[:, 2:]
    data[data == "NR"] = 0
    raw_data = data.to_numpy()
    return raw_data


def get_test_label(raw_data):
    month_data = {}
    for month in range(12):
        sample = np.empty([18, 180])
        for day in range(20):
            sample[:, day * 9:(day + 1) * 9] = raw_data[18 * (20 * month + day):18 * (20 * month + day + 1), :]
        month_data[month] = sample

    x = np.empty([12 * 20, 18 * 9], dtype=float)
    for month in range(12):
        for day in range(20):
            x[month * 20 + day, :] = month_data[month][:, day * 9:day * 9 + 9].reshape(
                1, -1)
    if not Config.device:
        x = torch.tensor(normalization(x, rangex), dtype=torch.float32)
    else:
        x = torch.tensor(normalization(x, rangex), dtype=torch.float32).cuda()
    return x
