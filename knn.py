import numpy as np
import matplotlib.pyplot as plt

def load():
    return np.load('dataset/data1.npz')

if __name__=="__main__":
    data = load()

    print("X Data shape: ", data['X'].shape)
    print("y Data shape: ", data['y'].shape)

    plt.scatter(data['X'], data['y'])
    plt.show()