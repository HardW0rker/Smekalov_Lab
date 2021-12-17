import numpy as np

def Calculate_the_function(x,w,b): #Вычисление функции
    sum = 0
    for i in range(0,len(x)):
        sum = sum + x[i]*w[i]
    return sum+b

if __name__ == '__main__':
    list_vec1 = [5,2,3] 
    lict_vec2 = [2,1,2]
    x = np.array(list_vec1) #Создание вектора x
    w = np.array(lict_vec2) #Создание вектора w
    b = 5
    sum = Calculate_the_function(x,w,b)
    print(sum)

