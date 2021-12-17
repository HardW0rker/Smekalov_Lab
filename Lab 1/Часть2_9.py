import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Пункт а

    norm1 = np.random.normal(size=100) #Генерация 100 значений из нормального распределения
    bin_numb = 10 
    plt.ylabel('Частота')
    plt.xlabel('Данные')

    #Отрисовка гистограммы
    plt.hist(norm1, density=True, bins = bin_numb)
    x = np.linspace(-5.5, 5.5, num=100)
    plt.plot(x, stats.norm.pdf(x, 0, 1))
    plt.show()
    
    print("Среднее = ",np.mean(norm1),"при исходном 0")
    print("Отклонение = ",np.std(norm1),"при исходном 1")

    # Пункт б

    g_dispN = 0
    cor_g_dispN  = 0
    l_dispN = 0
    cor_l_dispN = 0
    mseN = 0
    cor_mseN = 0

    for i in range(0,150):
        norm2 = np.random.normal(size=20)
        dispN = np.var(norm2,ddof=0)
        cor_dispN = np.var(norm2,ddof=1)
        
        if (dispN > 1):
            g_dispN += 1 #Количество раз когда дисперсия превысила реальную
        if (cor_dispN > 1):
            cor_g_dispN += 1 #Количество раз когда исправленная дисперсия превысила реальную
        if (dispN < 1):
            l_dispN += 1 #Количество раз когда дисперсия недооценила реальную
        if (cor_dispN < 1):
            cor_l_dispN += 1 #Количество раз когда исправленная дисперсия недооценила реальную
        
        mseN = (mseN+(1-dispN)**2)/(i+1) #Средней квадрат ошибки дисперсии
        cor_mseN = (cor_mseN+(1-cor_dispN)**2)/(i+1) #Средней квадрат ошибки исправленной дисперсии

    print('Количество раз когда дисперсия превысила реальную: ', g_dispN)
    print('Количество раз когда исправленная дисперсия превысила реальную: ', cor_g_dispN)

    print('Количество раз когда дисперсия недооценила реальную: ', l_dispN)
    print('Количество раз когда исправленная дисперсия недооценила реальную: ', cor_l_dispN)

    print('Средней квадрат ошибки дисперсии', mseN)
    print('Средней квадрат ошибки исправленной дисперсии', cor_mseN)
