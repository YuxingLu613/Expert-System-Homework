# !/usr/bin/env python3
# encoding: utf-8

"""
@author: Colton Lu
@contact: yxlu@stu.suda.edu.cn
@file: create_sequential_data
@time: 2021/6/10 14:43
"""

from prepare_data import clean_train_data

sales_train_path = "./data/sales_train.csv"
sales_train_data = clean_train_data(sales_train_path)
