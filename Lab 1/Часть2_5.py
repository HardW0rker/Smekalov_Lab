import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    x = np.arange(-10, 10, 0.01)

    def function_sin(x): #Функция синуса
        return np.sin(x)
 
    plt.title("График функции")
    plt.plot(x, function_sin(x))#Вывод графика sin(x)
    plt.grid(True)
    plt.show()

    plt.title("Производная функции")
    for i in x:
        plt.plot(i,(function_sin(i)-function_sin(i-0.01))/0.01, 'r.')#Вывод графика производной sin(x) точками
    plt.grid(True)
    plt.show()

