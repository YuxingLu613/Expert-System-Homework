# !/usr/bin/env python3
# encoding: utf-8

"""
@author: Colton Lu
@contact: yxlu@stu.suda.edu.cn
@file: main
@time: 2021/6/8 16:02
"""

from prepare_data import clean_train_data, get_data_by_item_id, get_data_by_shop_id
from data_analysis import get_month_analysis, count_outdated_data, get_shop_analysis
from shop_information import get_shops, refactor_shops
import matplotlib.pyplot as plt
import numpy as np

sales_train_path = "./data/sales_train.csv"
sales_train_data_pd = clean_train_data(sales_train_path)
sales_by_item_id = get_data_by_item_id(sales_train_data_pd)
sales_by_shop_id = get_data_by_shop_id(sales_train_data_pd)
get_month_analysis(sales_by_item_id)
count_outdated_data(sales_by_item_id)
notexist, outdated = get_shop_analysis(sales_by_shop_id)

shops_path = './data/shops.csv'
shops = get_shops(shops_path)
shops = refactor_shops(shops)
