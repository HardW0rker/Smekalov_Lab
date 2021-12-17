import numpy as np
import matplotlib.pyplot as plt

def Calculate_the_function_f(x): #Вычисление функции f(x)
    return x - (x**3)/6 + (x**5)/120 - (x**7)/5040

def Calculate_the_function_g(x): #Вычисление функции g(x)
    return np.sin(x)

if __name__ == '__main__':
    xlist = np.linspace(-5.0, 5.0, 1000) 
    ylist_f = [Calculate_the_function_f(x) for x in xlist]
    ylist_g = [Calculate_the_function_g(x) for x in xlist]
    plt.plot(xlist, ylist_f)
    plt.plot(xlist, ylist_g)
    plt.show()
