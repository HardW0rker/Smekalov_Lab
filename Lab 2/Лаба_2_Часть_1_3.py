import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'housing.csv')
df = pd.read_csv(file_path,sep = ',')
one_hot = pd.get_dummies(df['ocean_proximity'])
df = df.drop('ocean_proximity',axis = 1)
df = df.join(one_hot)

def avg_bedrooms(row):
    row['total_bedrooms']= row['total_bedrooms']/row['households']
    return row

df = df.apply(avg_bedrooms,axis=1)

def avg_rooms(row):
    row['total_rooms']= row['total_rooms']/row['households']
    return row

df = df.apply(avg_rooms,axis=1)
df = df.rename({'total_bedrooms': 'average_bedrooms', 'total_rooms': 'average_rooms'}, axis=1) 

df_train_val, df_test = train_test_split(df, test_size=0.1)

df_train, df_val = train_test_split(df_train_val, test_size = 0.33)
print("Количесвто строк и полей, в тренеровочной выборке", df_train.shape)
print("Количесвто строк и полей, в валидационной выборке", df_val.shape)

print("Число экземпляров данных, для которых признак average_bedrooms отсутствует.", df['average_bedrooms'].isnull().sum())

print("Пропущенных в тестовой",df_test['average_bedrooms'].isnull().sum()) 
print("Пропущенных в валидационной",df_val['average_bedrooms'].isnull().sum()) 
print("Пропущенных в обучающей",df_train['average_bedrooms'].isnull().sum()) 


snsplot = sns.kdeplot(df['average_bedrooms'], shade=True)
fig = snsplot.get_figure()

sns.set()
plt.show()

def metric(df, column):
    mn = df[column].mean()
    st = df[column].std()
    return [mn,st]
    
def row_correct(row, mn,st, column):
    if row.isnull().sum()!=0:
        row[column] = np.random.normal(mn,st)
    return row

def fill_gaps(df, column):
    mn, st = metric(df, column)
    df = df.apply(lambda x: row_correct(x,mn,st, column), axis=1)
    return df
    
column = 'average_bedrooms'
df_val = fill_gaps(df_val, column)
df_train = fill_gaps(df_train, column)
df_test = fill_gaps(df_test, column)

print("Пропущенных в тестовой ", df_test['average_bedrooms'].isnull().sum()) 
print("Пропущенных в валидационной ", df_val['average_bedrooms'].isnull().sum()) 
print("Пропущенных в обучающей", df_train['average_bedrooms'].isnull().sum())

def row_normalize(row, mn, st, column):
    row[column] = (row[column]-mn)/st
    return row

def df_normalize(df, columns):    
    for column in columns:
        mn,st = metric(df, column)
        df = df.apply(lambda x: row_normalize(x,mn,st, column), axis=1)
    return df
    
df_test = df_normalize(df_test,['longitude', 'latitude'])
df_val = df_normalize(df_val,['longitude', 'latitude'])
df_train = df_normalize(df_train,['longitude', 'latitude'])

def print_mn_st(df, cols):
    for c in cols:
        print('Среднее ', c, ':',  df[c].mean())
        print('Дисперсия ', c, ':',df[c].std())
        
cols = ['longitude', 'latitude']

print_mn_st(df_test, cols)
print_mn_st(df_val, cols)
print_mn_st(df_train, cols)





