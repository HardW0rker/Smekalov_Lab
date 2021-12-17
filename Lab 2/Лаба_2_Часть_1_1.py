import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    path_ord = os.path.join(script_dir, 'orders.csv')
    path_prod = os.path.join(script_dir, 'products.csv')
    colnames_ord=['OrderID', 'OrderDate', 'ProductID', 'UnitPrice', 'Quantity', 'Discount']
    colnames_prod=['ProductID', 'ProductName', 'QuantityPerUnit', 'UnitCost', 'UnitPrice', 'CategoryName']
    ord = pd.read_csv(path_ord, names=colnames_ord, header=None)
    prod = pd.read_csv(path_prod, names=colnames_prod, header=None)

    #Пункт 1
    Category = {}
    CategoryName = prod['CategoryName'][1]
    Category[CategoryName] = '0'


    for i in range(1, len(prod['CategoryName'])):
        if(prod['CategoryName'][i] == CategoryName):
            Category[CategoryName] = str(int(Category[CategoryName]) +   1)
        else:
            Category[prod['CategoryName'][i]] = '1'
            CategoryName = prod['CategoryName'][i]
    print("Число уникальных продуктов в каждой категории")
    print(Category)

    #Пункт 2
    NameSeafood = []
    for i in range(1, len(prod['CategoryName'])):
        if(prod['CategoryName'][i] == 'Морепродукты'):
            NameSeafood.append(prod['ProductName'][i])
    print("Все продукты в категории Морепродукты")        
    print(NameSeafood)

    #Пункт 3
    Date = {}
    for i in range(1, len(ord['OrderDate'])):
        OrderDate = ord['OrderDate'][i]
        Date[OrderDate[:7]] =  '0'
    for i in range(1, len(ord['OrderDate'])):
        OrderDate = ord['OrderDate'][i]
        Date[OrderDate[:7]] = str(int(Date[OrderDate[:7]]) + int(ord['Quantity'][i]))
    
    list_keys = list(Date.keys())
    list_keys.sort()
    dates = []
    values_date = []
    for i in list_keys:
        dates.append(i)
        values_date.append(int(Date[i]))
    
    plt.figure(figsize=(20,10))
    xVals = range(len(Date))
    plt.plot(xVals, values_date)
    plt.xticks(xVals, dates, fontsize=3.5)
    plt.show()

    #Пункт 4
    ord = ord.assign(OrderSum = lambda x:'')
    ord['OrderSum'][0] = 'OrderSum' 
    for i in range(1, len(ord['OrderSum'])):
        ord['OrderSum'][i] = (float(ord['UnitPrice'][i]) * float(ord['Quantity'][i]) * (1 - float(ord['Discount'][i])))
    print(ord)
    Sum_orders = {}
    for i in range(1, len(ord['OrderID'])):
        Sum_orders[ ord['OrderID'][i]] =  '0'
    for i in range(1, len(ord['OrderID'])):
        Sum_orders[ord['OrderID'][i]] = float(Sum_orders[ord['OrderID'][i]]) + float(ord['OrderSum'][i])
    sorted_orders = sorted(Sum_orders.values())

    print("10 самых дорогих заказов")
    for i in range(len(sorted_orders)-1, len(sorted_orders)-9, -1):
        print("OrderID: ",list(Sum_orders.keys())[list(Sum_orders.values()).index(sorted_orders[i])] , "Сумма заказа: ", sorted_orders[i])
    
    #Пункт 5
    Product_Cost = {}
    for i in range(1, len(prod['ProductName'])):
        Product_Cost[ prod['ProductName'][i]] =  '0'
    for i in range(1, len(prod['ProductName'])):
        Product_Cost[prod['ProductName'][i]] = float(Product_Cost[prod['ProductName'][i]]) + (float(prod['UnitPrice'][i]) / float(prod['QuantityPerUnit'][i]))
    sorted_product = sorted(Product_Cost.values())
    print("10 самых дорогих продуктов за шт")
    for i in range(len(sorted_product)-1, len(sorted_product)-9, -1):
        print(f"{'Продукт: ' + list(Product_Cost.keys())[list(Product_Cost.values()).index(sorted_product[i])]:<25} Стоимость продукта: {sorted_product[i]}")
        