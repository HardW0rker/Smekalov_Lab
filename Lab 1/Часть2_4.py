import numpy as np
import matplotlib.pyplot as plt
import pylab

def transformation_plot(plots, matrix): #Преобразование точек при помощи матрицы
    for i in range(len(plots)):
        plots[i][0] =  plots[i][0] * matrix[0][0] + plots[i][1] * matrix[0][1]
        plots[i][1] =  plots[i][0] * matrix[1][0] + plots[i][1] * matrix[1][1]
    return plots


if __name__ == '__main__':

    matrix = [[2,4],[1,5]] #Матрица
    plots = [[1,3],[0,2],[4,-1],[-6,9]] #Двумерные точки

    pylab.subplot (2, 1, 1)

    for i in range(len(plots)): #Отрисовка изначальных координат точек
        plt.plot(plots[i][0], plots[i][1],'ro') 
        plt.annotate(str(i+1), xy=(plots[i][0], plots[i][1]),xytext=(plots[i][0]+0.1,plots[i][1]+0.1)) #Номер точки
        pylab.title ("Начальное значение точек")

    plots = transformation_plot(plots, matrix) #Изменение координат точек

    for i in range(len(plots)):#Отрисовка изменённых координат точек
        pylab.subplot (2, 1, 2)
        plt.plot(plots[i][0], plots[i][1],'ko')
        plt.annotate(str(i+1), xy=(plots[i][0], plots[i][1]),xytext=(plots[i][0]+0.2,plots[i][1]+0.2)) #Номер точки
        pylab.title ("Преобразованное значение точек")

    plt.subplots_adjust(hspace=0.5)
    plt.show()