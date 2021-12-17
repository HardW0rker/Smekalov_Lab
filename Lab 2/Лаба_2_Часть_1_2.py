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
    ord = ord.assign(OrderSum = lambda x:'')
    ord['OrderSum'][0] = 'OrderSum' 
    for i in range(1, len(ord['OrderSum'])):
        ord['OrderSum'][i] = (float(ord['UnitPrice'][i]) * float(ord['Quantity'][i]) * (1 - float(ord['Discount'][i])))

    Category = {}
    for i in range(1, len(prod['CategoryName'])):
        Category[prod['CategoryName'][i]] = 0
    for i in range(1, len(prod['CategoryName'])):
        Category[prod['CategoryName'][i]] =  Category[prod['CategoryName'][i]] + 1
    
    Sum_Id = {}
    for i in range(1, len(ord['ProductID'])):
        Sum_Id[ ord['ProductID'][i]] =  0
    for i in range(1, len(ord['ProductID'])):
        Sum_Id[ord['ProductID'][i]] =  float(Sum_Id[ord['ProductID'][i]]) + float(ord['OrderSum'][i])

    Category_sum = {}
    for i in range(1, len(prod['CategoryName'])):
        Category_sum[ prod['CategoryName'][i]] = 0
    for i in range(1, len(prod['CategoryName'])):
        Category_sum[prod['CategoryName'][i]] = Category_sum[prod['CategoryName'][i]] + Sum_Id[prod['ProductID'][i]]

    Category_average_value = {k: Category_sum[k] / Category[k] for k in Category if k in Category_sum}
    print("Средний доход от продаж каждой категории: ")
    print(Category_average_value)

    #Пункт 2
    Purchase_price_Id = {}
    for i in range(1, len(prod['ProductID'])):
        Purchase_price_Id[ prod['ProductID'][i]] =  0
    for i in range(1, len(prod['ProductID'])):
        Purchase_price_Id[prod['ProductID'][i]] = float(prod['UnitCost'][i])

    Purchase_price_Sum = {}
    for i in range(1, len(ord['ProductID'])):
        Purchase_price_Sum[ ord['ProductID'][i]] =  0
    for i in range(1, len(ord['ProductID'])):
        Purchase_price_Sum[ord['ProductID'][i]] = Purchase_price_Sum[ord['ProductID'][i]] + (Purchase_price_Id[ord['ProductID'][i]] * float(ord['Quantity'][i]))

    Profit_values = {k: Sum_Id[k] - Purchase_price_Sum[k] for k in Purchase_price_Sum if k in Sum_Id}

    prod = prod.assign(Profit = lambda x:'')
    prod['Profit'][0] = 'Profit' 
    for i in range(1, len(prod['Profit'])):
        prod['Profit'][i] = Profit_values[prod['ProductID'][i]]
    print(prod)

    #Пункт 3

    Sum_Id_2005_2006 = {}
    Purchase_price_Sum_2005_2006 = {}
    for i in range(1, len(ord['ProductID'])):
        Sum_Id_2005_2006[ ord['ProductID'][i]] =  0
        Purchase_price_Sum_2005_2006[ ord['ProductID'][i]] =  0
    for i in range(1, len(ord['ProductID'])):
        OrderDate = ord['OrderDate'][i]
        if(OrderDate[:4] == '2005' or OrderDate[:4] == '2006'):
            Sum_Id_2005_2006[ord['ProductID'][i]] =  float(Sum_Id_2005_2006[ord['ProductID'][i]]) + float(ord['OrderSum'][i])
            Purchase_price_Sum_2005_2006[ord['ProductID'][i]] = Purchase_price_Sum_2005_2006[ord['ProductID'][i]] + (Purchase_price_Id[ord['ProductID'][i]] * float(ord['Quantity'][i]))
    
    Profit_values_2005_2006 = {k: Sum_Id_2005_2006[k] - Purchase_price_Sum_2005_2006[k] for k in Purchase_price_Sum_2005_2006 if k in Sum_Id_2005_2006}

    Category_Profit_2005_2006 = {}
    for i in range(1, len(prod['CategoryName'])):
        Category_Profit_2005_2006[ prod['CategoryName'][i]] = 0
    for i in range(1, len(prod['CategoryName'])):
        Category_Profit_2005_2006[prod['CategoryName'][i]] = Category_Profit_2005_2006[prod['CategoryName'][i]] + Profit_values_2005_2006[prod['ProductID'][i]]

    sorted_Category_Profit_2005_2006 = sorted(Category_Profit_2005_2006.values())

    sum_Profit_values_2005_2006 = 0
    for i in range(1, len(Profit_values_2005_2006)):
        sum_Profit_values_2005_2006 = sum_Profit_values_2005_2006 + Profit_values_2005_2006[prod['ProductID'][i]]

    sorted_sum = 0
    print('Наибольшая суммарная прибыль товаров категорий за 2005-2006 год, состовляющая 80% общей прибыли за этот период')
    for i in range(len(sorted_Category_Profit_2005_2006)-1, len(sorted_Category_Profit_2005_2006)-9, -1):
        print(f"{'Категория: ' + list(Category_Profit_2005_2006.keys())[list(Category_Profit_2005_2006.values()).index(sorted_Category_Profit_2005_2006[i])]:<35} Прибыль за категорию: {sorted_Category_Profit_2005_2006[i]}")
        sorted_sum += sorted_Category_Profit_2005_2006[i]
        if(sorted_sum >= sum_Profit_values_2005_2006*0.8):
            break

    Sum_Id = {}
    Purchase_price_Sum = {}
    for i in range(1, len(ord['ProductID'])):
        Sum_Id[ ord['ProductID'][i]] =  0
        Purchase_price_Sum[ ord['ProductID'][i]] =  0
    for i in range(1, len(ord['ProductID'])):
        OrderDate = ord['OrderDate'][i]
        Sum_Id[ord['ProductID'][i]] =  float(Sum_Id[ord['ProductID'][i]]) + float(ord['OrderSum'][i])
        Purchase_price_Sum[ord['ProductID'][i]] = Purchase_price_Sum[ord['ProductID'][i]] + (Purchase_price_Id[ord['ProductID'][i]] * float(ord['Quantity'][i]))
    
    Profit_values = {k: Sum_Id[k] - Purchase_price_Sum[k] for k in Purchase_price_Sum if k in Sum_Id}

    Category_Profit = {}
    for i in range(1, len(prod['CategoryName'])):
        Category_Profit[ prod['CategoryName'][i]] = 0
    for i in range(1, len(prod['CategoryName'])):
        Category_Profit[prod['CategoryName'][i]] = Category_Profit[prod['CategoryName'][i]] + Profit_values[prod['ProductID'][i]]

    sorted_Category_Profit = sorted(Category_Profit.values())

    sum_Profit_values = 0
    for i in range(1, len(Profit_values)):
        sum_Profit_values = sum_Profit_values + Profit_values[prod['ProductID'][i]]

    sorted_sum = 0
    print('Наибольшая суммарная прибыль товаров категорий за все года, состовляющая 80% общей прибыли за этот период')
    for i in range(len(sorted_Category_Profit)-1, len(sorted_Category_Profit)-9, -1):
        print(f"{'Категория: ' + list(Category_Profit.keys())[list(Category_Profit.values()).index(sorted_Category_Profit[i])]:<35} Прибыль за категорию: {sorted_Category_Profit[i]}")
        sorted_sum += sorted_Category_Profit[i]
        if(sorted_sum >= sum_Profit_values*0.8):
            break