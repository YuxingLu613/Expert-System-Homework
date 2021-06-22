# !/usr/bin/env python3
# encoding: utf-8

"""
@author: Colton Lu
@contact: yxlu@stu.suda.edu.cn
@file: prepare_data
@time: 2021/6/8 16:03
"""

import pandas as pd
from pprint import pprint as pp
import numpy as np


def clean_train_data(sales_train_path):
    print("-----Cleaning Dataset")
    # clean the dataset according the analyze result
    sales_train_data_pd = pd.read_csv(sales_train_path)
    # drop those items price higher than 10000 and below 0
    sales_train_data_pd = sales_train_data_pd[
        (sales_train_data_pd.item_price < 10000) & (sales_train_data_pd.item_price > 0)]
    # drop those items sales number more than 1000/day
    sales_train_data_pd = sales_train_data_pd[sales_train_data_pd.item_cnt_day < 1000]
    # group by id
    sales_train_data_pd = sales_train_data_pd.groupby(["item_id", "shop_id", "date_block_num"]).sum().reset_index()
    # sales_train_data_pd = sales_train_data_pd.rename(index=str, columns={"item_cnt_day": "item_cnt_month"})
    # sales_train_data_pd = sales_train_data_pd[["item_id", "shop_id", "date_block_num", "item_cnt_month"]]
    # pp(sales_train_data_pd['date_block_num'].describe())
    # pp(sales_train_data_pd.info())
    return sales_train_data_pd


def get_data_by_item_id(sales_train_data_pd):
    # count in month unit
    sales_by_item_id = sales_train_data_pd.pivot_table(index=['item_id'], values=['item_cnt_day'],
                                                       columns='date_block_num', aggfunc=np.sum,
                                                       fill_value=0).reset_index()
    sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)
    sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)
    sales_by_item_id.columns.values[0] = 'item_id'
    # pp(sales_by_item_id)
    return sales_by_item_id


def get_data_by_shop_id(sales_train_data_pd):
    sales_by_shop_id = sales_train_data_pd.pivot_table(index=['shop_id'], values=['item_cnt_day'],
                                                       columns='date_block_num', aggfunc=np.sum,
                                                       fill_value=0).reset_index()
    sales_by_shop_id.columns = sales_by_shop_id.columns.droplevel().map(str)
    sales_by_shop_id = sales_by_shop_id.reset_index(drop=True).rename_axis(None, axis=1)
    sales_by_shop_id.columns.values[0] = 'shop_id'
    return sales_by_shop_id

