# !/usr/bin/env python3
# encoding: utf-8

"""
@author: Colton Lu
@contact: yxlu@stu.suda.edu.cn
@file: shop_information
@time: 2021/6/20 16:30
"""
import pandas as pd


def get_shops(shop_path):
    return pd.read_csv(shop_path)


def refactor_shops(shops):
    shops['shop_name'] = shops['shop_name'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+',
                                                                                                              '').str.strip()
    shops['shop_city'] = shops['shop_name'].str.partition(' ')[0]
    shops['shop_type'] = shops['shop_name'].apply(lambda
                                                      x: 'мтрц' if 'мтрц' in x else 'трц' if 'трц' in x else 'трк' if 'трк' in x else 'тц' if 'тц' in x else 'тк' if 'тк' in x else 'NO_DATA')
    return shops
