# !/usr/bin/env python3
# encoding: utf-8

"""
@author: Colton Lu
@contact: yxlu@stu.suda.edu.cn
@file: distribution analysis
@time: 2021/6/8 23:20
"""

import numpy as np
import pandas as pd
from sqlalchemy import true
from tqdm import tqdm
import matplotlib.pyplot as plt

sales_train_path = "sales_train.csv"
sales_train_data_pd = pd.read_csv(sales_train_path)
sales_train_data_np = np.array(sales_train_data_pd)

print("-----Analyze Item Price")
# analyze the distribution of prices
all_prices = [x[4] for x in tqdm(sales_train_data_np)]
y = [0] * len(all_prices)
plt.scatter(all_prices, y, alpha=0.6, s=100)
plt.legend(["item price", ""])
plt.savefig("distribution of item price")
plt.show()

    print("-----Analyze Item Cnt Day")
    # analyze the sales of item
    all_cnt = [x[5] for x in tqdm(sales_train_data_np)]
    y = [0] * len(all_cnt)
    plt.scatter(all_cnt, y, alpha=0.6, s=100)
    plt.legend(["item cnt day", ""])
    plt.savefig("distribution of item cnt day")
    plt.show()

    print("-----Analyze Date Block")
    # analyze the distribution in date blocks
    all_blocks = [x[1] for x in tqdm(sales_train_data_np)]

    plt.hist(all_blocks, bins=max(all_blocks) + 1, rwidth=0.7)
    plt.xlabel("block")
    plt.legend(["records"])

    plt.savefig("./analysis result/distribution of date block")
    plt.show()
