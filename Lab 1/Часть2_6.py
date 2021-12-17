import numpy as np
import random

def Calculate_the_angle(v,q): #Определение угла между векторами v и q
    v_u = (q / np.linalg.norm(v))
    q_u = (v / np.linalg.norm(q))
    radians = np.arccos(np.clip(np.dot(v_u, q_u), -1.0, 1.0)) #Определение угла между векторами v и q в радианах
    angle = np.degrees([radians.real])[0] #Перевод угла из радиан в градусы
    return angle

if __name__ == '__main__':
    
    list_vec = []
    V = 1000 #Количество векторов
    d = 5 #Размерность пространства

    for i in range(V): #Создание случайных векторов
        temp_list = []
        for j in range(d):
            temp_list.append(random.randint(-100, 100)) #Создание вектора с координатами в пределах от -100 до 100
        list_vec.append(np.array(temp_list))

    temp_list = []
    for j in range(d): #Создание вектора q
            temp_list.append(random.randint(-100, 100)) #Создание вектора с координатами в пределах от -100 до 100
    q =  np.array(temp_list) 

    angle_90 = angle_30 = 0
    for i in range(V):
        angle = Calculate_the_angle(list_vec[i],q) #Определение угла между векторами
        if angle < 90:
            angle_90 += 1
        if angle < 30:
            angle_30 += 1
            
    print(angle_90/V)
    print(angle_30/V)