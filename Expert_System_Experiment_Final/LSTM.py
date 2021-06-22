# !/usr/bin/env python3
# encoding: utf-8

"""
@author: Colton Lu
@contact: yxlu@stu.suda.edu.cn
@file: LSTM
@time: 2021/6/22 0:17
"""
from prepare_data import clean_train_data, get_data_by_item_id, get_data_by_shop_id
from data_analysis import get_month_analysis, count_outdated_data, get_shop_analysis
from shop_information import get_shops, refactor_shops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

sales_train_path = "./data/sales_train.csv"
test_data = pd.read_csv("./data/test.csv")
sales_train_data_pd = pd.read_csv(sales_train_path)
sales_train_data_pd['date'] = pd.to_datetime(sales_train_data_pd['date'], format='%d.%m.%Y')
dataset = sales_train_data_pd.pivot_table(index=['shop_id', 'item_id'], values=['item_cnt_day'],
                                          columns=['date_block_num'], fill_value=0, aggfunc='sum')
dataset.reset_index(inplace=True)
dataset = pd.merge(test_data, dataset, on=['item_id', 'shop_id'], how='left')
dataset.fillna(0, inplace=True)
dataset.drop(['shop_id', 'item_id', 'ID'], inplace=True, axis=1)
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
y_train = dataset.values[:, -1:]
X_test = np.expand_dims(dataset.values[:, 1:], axis=2)
print(X_train,y_train,X_test.shape)

my_model = Sequential()
my_model.add(LSTM(units=64, input_shape=(33, 1)))
my_model.add(Dropout(0.4))
my_model.add(Dense(1))

my_model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
my_model.summary()
my_model.fit(X_train, y_train, batch_size=128, epochs=10)


submission_pfs = my_model.predict(X_test)
submission_pfs = submission_pfs.clip(0, 20)
submission = pd.DataFrame({'ID': test_data['ID'], 'item_cnt_month': submission_pfs.ravel()})
submission.to_csv('submission.csv', index=False)
