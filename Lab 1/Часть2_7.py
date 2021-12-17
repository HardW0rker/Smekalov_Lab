import numpy as np
import random
import matplotlib.pyplot as plt

def Checking_circle(list_points,a,d,n_points): #Определение отношения объёма гиперсферы к объёму гиперкуба
    sum_cercle = 0
    for i in range(n_points):
        sum = 0
        for j in range(d):
            sum = sum + (list_points[i][j])**2 
        if sum <= (a/2)**2: #Попадает ли точка в вписанную окружность
            sum_cercle +=1
    return sum_cercle/n_points
        

if __name__ == '__main__':
    list_points = []
    a = 100 #Сторона куба
    n_points = 10000 # Колличесвто точек

    dmin = 1 # Не может быть меньше 1
    dmax = 20 # Максимальная размерность
    xlist = np.linspace(dmin, dmax, dmax-dmin+1) 
    ylist = []

    
    for d in range(dmin,dmax+1):
        list_points = []
        for i in range(n_points):
            temp_points = []
            for j in range(d):
                temp_points.append(random.randint(-a/2, a/2)) #Создание точки с координатами от -сторона куба/2 до сторона куба/2
            list_points.append(temp_points)
        ylist.append(Checking_circle(list_points,a,d,n_points)) #Определение отношения объёма гиперсферы к объёму гиперкуба для разных размерностей
    
    plt.plot(xlist, ylist)
    plt.show()
 