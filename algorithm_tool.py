#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyecharts.charts as cht 
from pyecharts import options as opts

import numpy as np
import os
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten, Embedding, Activation, LeakyReLU
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[26]:


def file_sum(file_list):
    all_file = pd.DataFrame()
    for i in range(len(file_list)):
        file_df = pd.read_csv(file_list[i])
        all_file = pd.concat([all_file, file_df])
    # 필요한 컬럼만 추출
    all_file = all_file[["BKG_DATE", "ITEM_CD", "ITEM_QTY", 'CNEE_ADDR_1']]
    # 날짜순으로 정렬
    all_file = all_file.sort_values(by=["BKG_DATE"], axis=0)
    return all_file

def city_ratio(file_list, item_name):
    # 특정 상품의 행만 추출
    file = file_sum(file_list)
    file = file.loc[file['ITEM_CD'] == item_name]
    file = file[['ITEM_QTY', 'CNEE_ADDR_1']]
    # 1차 지역별 합치기
    city_item = file.groupby('CNEE_ADDR_1', as_index=False)
    city_item = city_item.sum()
    # 2차 지역별 합치기
    city_item['CNEE_ADDR_1'] = city_item['CNEE_ADDR_1'].str.slice(start=0, stop=2)
    city_item.loc[((city_item.CNEE_ADDR_1 == '경남') | (city_item.CNEE_ADDR_1 == '경북')), ('CNEE_ADDR_1')] = '경상'
    city_item.loc[((city_item.CNEE_ADDR_1 == '전남') | (city_item.CNEE_ADDR_1 == '전북')), ('CNEE_ADDR_1')] = '전라'
    city_item.loc[((city_item.CNEE_ADDR_1 == '충남') | (city_item.CNEE_ADDR_1 == '충북')), ('CNEE_ADDR_1')] = '충청'
    city_item = city_item.groupby('CNEE_ADDR_1', as_index=False)
    city_item = city_item.sum()
    # 지역과 상품 수량 합 리스트로 변환
    city = city_item['CNEE_ADDR_1'].values.tolist()
    item = city_item['ITEM_QTY'].values.tolist()
    # 그래프 그리기
    pie = cht.Pie()  
    pie.add("상품 갯수",[list(z) for z in zip(city, item)], radius=200, label_opts=opts.LabelOpts(formatter = "{b} : {c}({d}%)"))
    return display(pie.render())

def z_normalization(value1) :
    value1_mean = value1.mean(axis=0)
    value1_std = value1.std(axis=0)    
    value1 = (value1 - value1_mean) / value1_std
    return value1,value1_mean,value1_std

def z_return(value1,value2,value3) :
    value1 = value1*value3+value2
    value = np.array(list(map(np.int,value1)))
    value = value[108:]
    print(value)
        
    return value
            
def max_min_normalization(value) :
    value_max = max(value)
    value_min = min(value)    
    value = value - value_min
    return value

def make_dataset(dataset) :
    data = []
    labels = []
    future_target = 7
    start_index = 0 + future_target
    end_index = len(dataset)-6
    for i in range(start_index,end_index) :
        data.append(dataset[i-future_target:i])
        labels.append(dataset[i:i+7])
    data = np.array(data)
    data = data.reshape(data.shape[0],data.shape[1],1)
    return data, np.array(labels)

def make_testset(dataset) :
    future_target = 7
    data = dataset[:len(dataset)-2*future_target]
    x_test = []
    start_index = 0 + future_target
    end_index = len(dataset)-2*future_target
    for i in range(start_index,end_index) :
        x_test.append(data[i-future_target:i])
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
    return x_test

def predict_model(x_train,y_train,x_test,z_item) :
    model = tf.keras.Sequential()
    
    model.add(Conv1D(16,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2]),kernel_size = 2,strides = 1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size = 2))
    
    model.add(Conv1D(32,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2]),kernel_size = 2,strides = 1))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.GRU(units = 15,input_shape=(x_train.shape[1],x_train.shape[2]),activation = 'tanh'))
    model.add(tf.keras.layers.Dense(7))
    model.compile(loss='mse',optimizer='Nadam')
    
    history = model.fit(x_train,y_train,
                       batch_size = 1,
                       epochs = 20)
    
    first_pred = model.predict(x_test)
    days_7 = first_pred[len(first_pred)-1].tolist()
    first_pred = first_pred.reshape(first_pred.shape[0],first_pred.shape[1],1)
    second_pred = model.predict(first_pred)
    days_14 = second_pred[len(second_pred)-1].tolist() 
    pred = z_item[:108]
    real_pred = pred
    real_pred = real_pred.tolist()
    real_pred = real_pred + days_7 + days_14
    fig = plt.figure(facecolor='white',figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.plot(z_item, label = 'True')
    ax.plot(real_pred,label = 'Prediction')
    ax.legend()
    plt.show
    real_pred = np.array(real_pred)
    return real_pred

def item_prediction(file_list2, item_name) :
    # 특정 상품의 행만 추출
    file = file_sum(file_list2)
    file = file.loc[file['ITEM_CD'] == item_name]
    file = file[["BKG_DATE","ITEM_QTY"]]
    # 일 단위로 상품 개수 합
    daily_item = file.groupby("BKG_DATE", as_index=False)
    daily_item = daily_item.sum()
    # 날짜 0000-00-00 형식으로 변경
    daily_item["BKG_DATE"] = daily_item["BKG_DATE"].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    # 날짜를 인덱스로 
    daily_item = daily_item.set_index("BKG_DATE", drop = True, append = False, inplace = False)
    # 상품이 0개인 날짜를 추가한 후 개수 0개 부여
    daily_item = daily_item.asfreq('D', fill_value = 0)
    item = daily_item['ITEM_QTY'].values
    # 정규화
    z_item,z_item_mean,z_item_std= z_normalization(item)
#   max_min_item = max_min_normalization(item)
    # 학습용 데이터 생성
    x_train,y_train = make_dataset(z_item)
#   x_train,y_train = make_dataset(max_min_item)
    # 테스트 세트 생성
    x_test = make_testset(z_item)
#   x_test = make_testset(max_min_item)
    #학습하고 그래프 그리기 + 
    pred = predict_model(x_train,y_train,x_test,z_item)
    pred = z_return(pred,z_item_mean,z_item_std)
    item = item[108:]
    return pred,item
    
    


# In[ ]:




