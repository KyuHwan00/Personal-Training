# 변수가 객체의 참조값을 가진다는 것, 불변형 내부에도 가변형 컨테이너의 참조를 가질 수 있다는 것 등 / l1 = [1,332,44] / l2 = l1 id를 복사해온것 
# 참조로 인해서 발생할 수 있는 실수들이 많습니다.
# 얕은복사 copy() 깊은 복사 deepcopy() 얕은 복사와 깊은 복사 모두 객체를 새로 만들어서 각 변수가 참조하는 객체의 정체성(identity)는 다르다.
#하지만 얕은 복사에서는 객체 내부의 컨테이너가 참조하는 객체는 기존 변수의 객체 내부의 컨테이너 참조 값을 그대로 가져오므로, 
# 같은 정체성의 컨테이너를 참조하게 된다.
# 실수가 잦을 수 있는 주제이므로 한 번씩 확인해보면 좋을 것 같습니다.
# del 명령은 이름을 제거하는 것이지, 객체를 제거하는 것이 아니다. 변수 삭제 메모리에는 존재
# 가변형을 매개변수 기본값으로 사용하지말자 => 초기화 해주는 과정을 추가해야함, 변수 = None => if 변수 == None: self.변수 = [] (리스트는 가변형)
import copy
import numpy as np
from sklearn import datasets
class Singlelayer():
    def __init__(self, learing_rate = 0.1):
        self.w = None
        self.b = None
        self.loss = []
        self.w_history = []
        self.lr = learing_rate
    def forpass(self,x):
        return np.sum(self.w*x) + self.b
    def backprop(self, x, err):
        grad_w = -(err)*x
        grad_b = -(err)
        return grad_w, grad_b
    def activation(self, z):
        return 1/(1+np.exp(-z))
    def fit(self, x, y, epochs):
        self.w = np.ones(x.shape[1])
        self.b = 0
        self.w_history.append(copy.deepcopy(self.w))
        np.random.seed(42)
        for _ in range(epochs):
            loss= 0
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:
                y_hat = self.forpass(x[i])
                a = self.activation(y_hat)
                err = y[i] - a
                grad_w , grad_b = self.backprop(x[i], err)
                self.w -= self.lr*grad_w
                self.b -= self.lr*grad_b
                self.w_history.append(copy.deepcopy(self.w))
                self.loss.append(err)
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += -(y[i]*np.log(a)+ (1-y[i]*np.log(1-a)))
        self.loss.append(loss/len(y))
    def predict(self, x):
        z = [self.forpass(x_t) for x_t in x]
        return np.array(z) > 0
    def score(self, x,y):
        return np.mean(self.predict(x) == y)



from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
data = dataset.data
target = dataset.target

x_train , x_test , y_train , y_test = train_test_split(data, target, stratify= target, test_size= 0.2 , random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify= y_train, test_size=0.2, random_state=42)

sig = Singlelayer()
sig.fit(x_train, y_train,100)
sig.score(x_val, y_val)

import matplotlib.pyplot as plt

w2 = list()
w3 = list()

for w in sig.w_history:
    w2.append(w[2])
    w3.append(w[3])
plt.plot(w2, w3)
plt.plot(w2[-1], w3[-1], 'ro')
plt.xlabel('w2')
plt.ylabel('w3')
plt.show()

x_mean = np.mean(x_train, axis = 0)
x_std = np.std(x_train, axis = 0)
x_train_scaled = (x_train - x_mean) / x_std

sig = Singlelayer()
sig.fit(x_train_scaled, y_train,100)

w2 = list()
w3 = list()

for w in sig.w_history:
    w2.append(w[2])
    w3.append(w[3])
plt.plot(w2, w3)
plt.plot(w2[-1], w3[-1], 'ro')
plt.xlabel('w2')
plt.ylabel('w3')
plt.show()

# 특성의 범위가 다른경우, 즉 scale이 다른 경우에는 정규화를 통해서 scale을 조정해줘야한다
val_mean  = np.mean(x_val, axis =0)
val_std = np.std(x_val, axis =0)
x_val_scaled = (x_val - val_mean)/ val_std

sig = Singlelayer()
sig.fit(x_train_scaled, y_train,100)
sig.score(x_val_scaled, y_val)

# 위와 같이 훈련 세트와 검증세트를 각각 계산하면 기존의 훈련세트와 검증세트 사이의 거리가 유지가 되지않는다 / 즉 서로 다른 비율로 계산이됨
# 이걸 보완하기 위해서 검증세트의 x값을 훈련세트의 평균과 표준편차를 이용해서 정규화, 표준화를 시켜줌