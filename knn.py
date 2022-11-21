import numpy as np
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, data, indepent_num, k=20):
        self.indepent_num = indepent_num
        self.x = data['X'].reshape(-1, indepent_num)
        self.y = data['y'].reshape(-1, 1)
        self.k = k     
    
    def __count_distance(self, num):
        dist = num - self.x
        dist = np.sqrt(np.sum(np.square(dist), axis=1)).reshape(-1, 1)
        dist = np.hstack((dist, np.arange(len(self.x)).reshape(-1, 1)))
        return dist
    
    def __k_nearst(self, dist):
        return np.array(sorted(dist, key = lambda a: a[0])[:self.k])

    def __solve(self, index):
        index = index.astype("int32")
        x = np.hstack((self.x[index], np.ones((self.k, 1))))
        y = self.y[index]
        a = x.T.dot(x)
        b = x.T.dot(y)
        return np.linalg.solve(a, b)

    def predict(self, num):
        result = []
        num = num.reshape(-1, self.indepent_num)
        for n in num:
            neighborhood = self.__k_nearst(self.__count_distance(n))
            a = self.__solve(neighborhood[:, 1])
            pred = a[:-1].T.dot(n) + a[-1]
            result.append(pred[0])
        return np.array(result)
    
    def predict_plot2D(self, x_axis):
        y_axis = self.predict(x_axis)
        plt.figure()
        plt.scatter(self.x, self.y, label="training data")
        plt.scatter(x_axis, y_axis, color="red", label="prediction", s=3)
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.title("2D KNN Linear Regression (k=" + str(self.k) + ")")
        plt.show()
    
    def predict_plot3D(self, x_axis):
        y_axis = self.predict(x_axis)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(self.x[:, 0], self.x[:, 1], self.y, label="training data")
        ax.scatter(x_axis[:, 0],x_axis[:, 1], y_axis, color="red", linewidth=0, label="prediction")
        # ax.plot_trisurf(x_axis[:, 0],x_axis[:, 1], y_axis, color="red", linewidth=0, label="prediction")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")
        ax.set_title("3D KNN Linear Regression")
        plt.show()
            
if __name__=="__main__":
    #Load data
    data1 = np.load("dataset/data1.npz")
    #Create Model
    model1 = KNN(data1, 1)
    #Create predict data
    x_axis = np.arange(0, 10, 0.01)
    #show predict result
    model1.predict_plot2D(x_axis)
    
    #Load data
    data2 = np.load("dataset/data2.npz")
    #Create Model
    model2 = KNN(data2, 2)
    #Create predict data
    x_axis = []
    for x1 in range(-3, 4):
        for x2 in range(-3, 4):
            x_axis.append([x1, x2])
    x_axis = np.array(x_axis)
    #show predict result
    model2.predict_plot3D(x_axis)