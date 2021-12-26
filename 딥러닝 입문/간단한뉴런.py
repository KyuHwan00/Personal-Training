class Neuron:

    def __init__(self):
        self.w = 1.0
        self.b = 1.0
    def forpass(self,x):
        y_hat = self.w*x + self.b
        return y_hat

    def backpro(self, x, y):
        loss = -(y - self.forpass(x)) 
        grad_w = x*loss
        grad_b = 1*loss
        return grad_w, grad_b
    
    def fit(self,x ,y):
        for _ in range(100):
            for x_i, y_i in zip(x,y):
                grad_w , grad_b = self.backpro(x_i, y_i)
                self.w -= grad_w
                self.b -= grad_b
        return self.w , self.b

from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
diabetes = load_diabetes()

x = diabetes.data[:, 3]
y = diabetes.target


test = Neuron()
test.fit(x,y)


plt.scatter(x, y)
pt1 = (-0.1, -0.1*test.w + test.b)
pt2 = (0.15, 0.15*test.w + test.b)
plt.plot([pt1[0],pt2[0]], [pt1[1], pt2[1]])
plt.show()