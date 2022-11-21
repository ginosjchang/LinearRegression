import numpy as np
import matplotlib.pyplot as plt

data1 = np.load("dataset/data1.npz")
data2 = np.load("dataset/data2.npz")
test_data = dict()
# test_data['X'] = np.

class LW:
    def __init__(self, data, indepent_num, k = 10):
        self.x = data['X'].reshape(-1, indepent_num)
        self.y = data['y'].reshape(-1, 1)
        self.indepent_num = indepent_num
        self.k = k

        print("LW model parameters:")
        print("Indepent num: ", indepent_num, "\nk: ", k)
        print("training dataset:\n\tx shape: ", self.x.shape, "\n\ty shape: ", self.y.shape)
    
    def get_w(self, num):
        w = self.x - num
        w = np.exp(- w.dot(w.T)/(2 * self.k * self.k))
        return w

    def predict(self, num):
        print("Prediction")
        num = num.reshape(-1, self.indepent_num)
        result = []
        for n in num:
            w = self.get_w(n)
            theta = np.linalg.pinv(self.x.T.dot(w).dot(self.x)).dot(self.x.T).dot(w).dot(self.y)
            # print("n: ", n, " theta: ", theta, "result: ", np.dot(n, theta))
            result.append(np.dot(theta.T, n))
        
        return np.array(result)

if __name__=="__main__":
    model1 = LW(data1, 1, k=1)

    x_axis = np.arange(0, 11, 1)
    y_axis = model1.predict(x_axis)

    plt.figure()
    plt.scatter(data1['X'], data1['y'])
    plt.plot(x_axis, y_axis, color='red')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(data2['X'][:,0], data2['X'][:,1], data2['y'])
    # plt.show()