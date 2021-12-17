import os
import pandas as pd
import numpy as np
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, 'ccpp.csv')
df = pd.read_csv(csv_path)

df = df.apply(lambda x: x.str.replace(',','.')) # Изменяелм все , на . для того чтобы перевести во float

x = df.iloc[:, :4].to_numpy().astype(np.float32) # Переводим во float
y = df.iloc[:, 4].to_numpy().astype(np.float32) # Переводим во float

x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.33, shuffle=False)

model = SVR(kernel='linear')
model.fit(x_train, y_train)


#Для ввалидационной выборки
y_predict = model.predict(x_val)
mean_absolute_error(y_val, y_predict), r2_score(y_val, y_predict)
plt.figure(figsize=(20, 10))
plt.xlabel("Предсказание модели", fontsize=13)
plt.ylabel("Истинное значение", fontsize=13)
plt.title('Для вылидационной выбоки')
plt.plot([420, 500], [420, 500], c='r', label='t = y')
plt.scatter(y_predict, y_val)
plt.legend()
plt.show()

#Для тестовой выборки
y_predict_test = model.predict(x_test)
mean_absolute_error(y_test, y_predict_test), r2_score(y_test, y_predict_test)
plt.figure(figsize=(20, 10))
plt.xlabel("Предсказание модели", fontsize=13)
plt.ylabel("Истинное значение", fontsize=13)
plt.title('Для тестовой выбоки')
plt.plot([420, 500], [420, 500], c='r', label='t = y')
plt.scatter(y_predict_test, y_test)
plt.legend()
plt.show()






