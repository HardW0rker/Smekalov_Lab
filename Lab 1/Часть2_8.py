import numpy as np
import matplotlib.pyplot as plt


def matrix_tranform(matrix, N, d):
    matrix=np.transpose(matrix) #Транспонирование матрицы
    for i in range(0,d):
        mn=np.mean(matrix[i]) #Среднее значение в столбце
        st = np.std(matrix[i]) #Эмпирически подсчитанное стандартное отклонение в столбце
        for j in range(0,N):
            matrix[i][j]=(matrix[i][j]-mn)/st 
    matrix=np.transpose(matrix) #Транспонирование матрицы
    return matrix #Возвращаем изменённую матрицу
    
if __name__ == '__main__':
    
    N=7
    d=2
    matrix = np.zeros((N,d))

    for i in range(0,N):
        matrix[i]=np.random.multivariate_normal(mean=[1,2],cov=[[2,1],[1,3]]) #нормализуем матрицу

    print(matrix) #Выводим матрицу в консоль
        
    h = 1
    #Отрисовываем матрицу
    for i in range(0,N):
        plt.plot(matrix[i][0], matrix[i][1],'ro')
        plt.annotate(h,xy=(matrix[i][0]+0.1, matrix[i][1]+0.1))
        h+=1

    matrix=matrix_tranform(matrix, N, d) #Преобразование матрицы
    #Отрисовываем изменённую матрицу
    h = 1
    for i in range(0,N):
        plt.plot(matrix[i][0], matrix[i][1],'bo')
        plt.annotate(h,xy=(matrix[i][0]+0.1, matrix[i][1]+0.1))
        h+=1

    plt.show()
    print(matrix) #Выводим изменённую матрицу в консоль