import numpy as np
import matplotlib.pyplot as plt
import math

data1 = np.load("dataset/data1.npz")
data2 = np.load("dataset/data2.npz")

def plot(data):
    plt.scatter(data['X'], data['y'])
    plt.title("data1")
    plt.show()

class KNN:
    def __init__(self, data, indepent_num, k=3):
        self.train_data = data
        self.data_num = len(data['X'])
        self.indepent_num = indepent_num
        self.k = k     
    
    def count_distance(self, num):
        self.dist = []
        for i in range(self.data_num):
            self.dist.append([abs(num - self.train_data['X'][i]), self.train_data['y'][i]])
    
    def count_distance2(self, num):
        self.dist = []
        for i in range(self.data_num):
            diff = 0
            for j in range(self.indepent_num):
                diff += math.pow(num[j] - self.train_data['X'][i][j], 2)
            self.dist.append([math.sqrt(d), self.train_data['y'][i]])
    
    def k_nearst(self):
        return sorted(self.dist, key = lambda a: a[0])[:self.k]
        
    def predict(self, num):
        result = []
        for n in num:
            if self.indepent_num == 1:
                self.count_distance(n)
            else:
                self.count_distance(n)
            neighborhood = self.k_nearst()
            pred = 0
            for i in range(self.k):
                pred += neighborhood[i][1]
            result.append(pred / self.k)
        return result
        
if __name__=="__main__":
    model1 = KNN(data1, 1)

    x_axis = np.arange(0, 11, 0.5)
    y_axis = model1.predict(x_axis)
    
    plt.scatter(data1['X'], data1['y'])
    plt.plot(x_axis, y_axis, color="red")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("KNN linear regression")
    plt.show()

    # model2 = KNN(data2, 2)
    # fig = plt.figure()

    # ax = fig.gca(projection='3d')

    # ax.scatter(data2['X'][:,0], data2['X'][:,1], data2['y'])
    # plt.xlabel("X1")
    # plt.ylabel("X2")
    # plt.show()