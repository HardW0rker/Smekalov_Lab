import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    r = 100 #Радиус круга
    Circle = plt.Circle((0, 0), r, color='b')
    ax=plt.gca()
    ax.add_patch(Circle)
    plt.axis('scaled')
    plt.show()
