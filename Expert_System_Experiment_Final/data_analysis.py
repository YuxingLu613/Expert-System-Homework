# !/usr/bin/env python3
# encoding: utf-8

"""
@author: Colton Lu
@contact: yxlu@stu.suda.edu.cn
@file: data analysis
@time: 2021/6/20 16:00
"""
import matplotlib.pyplot as plt


def get_month_analysis(sales_by_item_id):
    # analyze the sum and mean by month range
    sales_by_item_id.sum()[1:].div(10000).plot(legend=True, label="Monthly sum")
    sales_by_item_id.mean()[1:].plot(legend=True, label="Monthly mean")
    plt.savefig("./analysis result/month analysis.png")


def count_outdated_data(sales_by_item_id):
    # count items that have no sales record in the last 6 months
    outdated_items = sales_by_item_id[sales_by_item_id.loc[:, '27':].sum(axis=1) == 0]
    print("outdated item number:", len(outdated_items))
    return len(outdated_items)


def get_shop_analysis(sales_by_shop_id):
    # not exist in fi
    not_exist = {}
    for i in range(6, 34):
        not_exist[i] = sales_by_shop_id['shop_id'][sales_by_shop_id.loc[:, '0':str(i)].sum(axis=1) == 0].unique()
    outdated = {}
    for i in range(6, 28):
        outdated[i] = sales_by_shop_id['shop_id'][sales_by_shop_id.loc[:, str(i):].sum(axis=1) == 0].unique()
    return not_exist, outdated
