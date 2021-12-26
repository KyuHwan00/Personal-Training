from numpy import iterable
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

diabetes = load_diabetes()

x = diabetes.data[:,3]
y = diabetes.target

plt.scatter(x,y)
plt.xlabel('diabetes.data')
plt.ylabel('diabetes.target')
plt.show()

w = 1.0
b = 1.0

for _ in range(10):
    for x_i, y_i in zip(x,y):
        y_hat = w*x_i + b
        loss = y_i - y_hat
        w_rate = x_i
        w, b = w + x_i*loss, b + 1*loss
xpoint = [x[0] , x[-1]]  
ypoint  = [x[0]*w +b, x[-1]*w +b]
plt.scatter(x,y)
plt.plot(xpoint, ypoint)
plt.xlabel('diabetes.data')
plt.ylabel('diabetes.target')
plt.show()

        