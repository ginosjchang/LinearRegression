import numpy as np
import matplotlib.pyplot as plt

data1 = np.load("dataset/data1.npz")
data2 = np.load("dataset/data2.npz")
test_data = dict()
# test_data['X'] = np.

class LW:
    def __init__(self, data, indepent_num, k = 1):
        self.x = data['X'].reshape(-1, indepent_num)
        self.y = data['y'].reshape(-1, 1)
        self.X = np.hstack((self.x, np.ones((len(self.x), 1))))
        self.indepent_num = indepent_num
        self.k = k

        # print("LW model parameters:")
        # print("Indepent num: ", indepent_num, "\nk: ", k)
        # print("training dataset:\n\tx shape: ", self.x.shape, "\n\ty shape: ", self.y.shape)
    
    def get_w(self, num):
        m = len(self.x)
        w = np.zeros((m, m))
        dist = np.sqrt(np.sum(np.square(self.x-num), axis=1))
        for i in range(m):
            w[i, i] = np.exp(dist[i]/(- 2 * self.k * self.k))
        return w

    def predict(self, num):
        num = num.reshape(-1, self.indepent_num)
        result = []
        for n in num:
            if self.indepent_num == 1:
                n = np.array(n).reshape(-1, self.indepent_num)
            w = self.get_w(n)
            theta = np.linalg.pinv(self.X.T.dot(w).dot(self.X)).dot(self.X.T).dot(w).dot(self.y)
            result.append((theta[:-1].T.dot(n) + theta[-1])[0])
        # print(result)
        return np.array(result)
    
    def predict_plot2D(self, x_axis):
        y_axis = self.predict(x_axis)
        plt.figure()
        plt.scatter(data1['X'], data1['y'], label="training")
        plt.scatter(x_axis, y_axis, color='red', label="prediction", s=3)
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.title("2D Locally weighted Regression")
        plt.show()
    
    def predict_plot3D(self, x_axis):
        y_axis = self.predict(x_axis)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(self.x[:, 0], self.x[:, 1], self.y, label="training data")
        ax.scatter(x_axis[:, 0],x_axis[:, 1], y_axis, color="red", linewidth=0, label="prediction")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")
        ax.set_title("3D Locally weighted Regression")
        plt.show()

if __name__=="__main__":
    #Load data
    data1 = np.load("dataset/data1.npz")
    #Create Model
    model1 = LW(data1, 1)
    #Create predict data
    x_axis = np.arange(0, 10, 0.01)
    #show predict result
    model1.predict_plot2D(x_axis)

    #Load data
    data2 = np.load("dataset/data2.npz")
    #Create Model
    model2 = LW(data2, 2)
    #Create predict data
    x_axis = []
    for x1 in range(-3, 4):
        for x2 in range(-3, 4):
            x_axis.append([x1, x2])
    x_axis = np.array(x_axis)
    #show predict result
    model2.predict_plot3D(x_axis)
    