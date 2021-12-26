import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = load_breast_cancer()
x = data.data
y = data.target

x_train , x_test , y_train ,y_test = train_test_split(x,y , test_size= 0.2, stratify= y , random_state= 42)
# 로지스틱 손실 함수의 미분 값을 미리 계산해서 구현시 사용, 한계 오직 로지스틱 회귀 문제만 풀 수 있음, 다른 softmax 또는 회귀 문제 시 값이 달라질 수 있음
class Singlelayer():
    def __init__(self): # 가중치와 편향 초깃값, losses 그래프를 그리기위해 설정
        self.w = None
        self.b = None
        self.losses = []
    def activation(self, z): # 활성화 함수로 시그모이드 함수를 사용, 미분 시 값이 sigmoid(x)*(1-sigmoid(x))이므로 계산하기 쉬움
        a = 1/(1+np.exp(-z))
        return a
    def forpass(self, x):# 정방향계산 np.sum 을 사용하는 이유는 각 특성의 결과값을 출력층에 전달하기 위해 합함
        z = np.sum(self.w*x) + self.b
        return z
    def backprop(self, x, y, a):# 손실함수를 활성화함수에 대해 미분 x 활성화함수를 z에 대해 미분 x z를 w에 대해 미분 => -(y-a)xi => 오차x입력
        grad_w = -(y-a)*x
        grad_b = -(y-a)
        return grad_w, grad_b
    def fit(self, x, y, epochs):
        self.w = np.ones(x.shape[1])
        self.b = 1
        
        for _ in range(epochs):
            loss = 0
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                grad_w , grad_b = self.backprop(x[i], y[i], a)
                self.w -= grad_w
                self.b -= grad_b
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += -(y[i]*np.log(a) +  (1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y))
    def predict(self,x_t):
        z = [self.forpass(x_i) for x_i in x_t]
        return np.array(z)>0
    def score(self,x,y):
        return np.mean(self.predict(x) == y)
layer = Singlelayer()
layer.fit(x_train,y_train, 100)
layer.score(x_test, y_test)

plt.plot(layer.losses)
plt.show()
plt.close()
