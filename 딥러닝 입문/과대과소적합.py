from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class SingleLayer():
    def __init__(self, learning_rate = 0.1):
        self.w = None
        self.b = None
        self.losses = list()
        self.w_history = list()
        self.val_losses = list()
        self.lr = learning_rate

    def forpass(self, x):
        z = np.sum(self.w*x) + self.b
        return z


    def activation(self, z):
        a = 1/(1+np.exp(-z))
        return a 

    def backprop(self, x, err):
        grad_w = err*x
        grad_b = err
        return grad_w , grad_b


    def fit(self, x, y, epochs, x_val= None, y_val= None):
        self.w = np.ones(x.shape[1])
        self.b = 0
        self.w_history.append(self.w.copy())
        np.random.seed(42)
        for _ in range(epochs):
            loss = 0
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                grad_w, grad_b = self.backprop(x[i], err)
                self.w -= self.lr * grad_w
                self.b -= grad_b
                self.w_history.append(self.w.copy())
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += -(y[i]*np.log(a) + (1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y))
            self.update_val_loss(x_val, y_val)


    def predict(self, x):
        z = [self.forpass(x_t) for x_t in x]
        return np.array(z) > 0

    def score(self, x, y):
        return np.mean(self.predict(x) == y)

    def update_val_loss(self, x_val, y_val):
        if x_val is None:
            return
        val_loss = 0
        for i in range(len(x_val)):
            z = self.forpass(x_val[i])
            a = self.activation(z)
            a = np.clip(a, 1e-10, 1-1e-10)
            val_loss += -(y_val[i]*np.log(a)+ (1-y_val[i])*np.log(1-a))
        self.val_losses.append(val_loss/len(y_val))
        

dataset = load_breast_cancer()
data = dataset.data
target = dataset.target

x_train_all , x_test , y_train_all , y_test = train_test_split(data, target, stratify= target, test_size= 0.2 , random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify= y_train_all, test_size=0.2, random_state=42)

x_mean = np.mean(x_train, axis = 0)
x_std = np.std(x_train, axis = 0)
x_train_scaled = (x_train - x_mean) / x_std
x_val_scaled = (x_val - x_mean)/ x_std


sig = SingleLayer()
sig.fit(x_train_scaled, y_train, 100, x_val= x_val_scaled, y_val= y_val)

plt.ylim(0, 0.3)
plt.plot(sig.losses)
plt.plot(sig.val_losses)
plt.legend(['train_loss', 'val_loss'])
plt.show()


layer2 = SingleLayer()
layer2.fit(x_train_scaled, y_train, 20)
layer2.score(x_val_scaled, y_val)


# 과대적합이란 훈련세트에서는 좋은 성능을 보이는 반면, 검증세트에서는 그렇지 못한 경우 => 즉 분산이 크다 , 모델 복잡도가 높다
# 다양한 훈련세트를 통해서 해결 혹은 가중치를 제한하여 모델의 복잡도를 낮춘다
# 과소적합은 훈련세트와 검증세트의 간격은 좁지만 전반적인 모델의 정확도가 낮은 경우를 말함 => 즉 편향이 크다 , 복잡도가 낮음
# 모델의 복잡도를 높이거나 가중치 제한을 완화
# 모델 복잡도란 모델이 가진 학습가능한 가중치 개수를 말함
# 과소와 과대 적합의 경우 편향-분산 트레이드 오프라는 관계를 가지는데, 하나가 커지면 다른 하나가 작아지는 관계이다
# 따라서 분산이나 편향 어느 한쪽이 너무 커지지 않도록 적절한 중간 지점을 선택 => 규제전에는 적절한 에포크를 찾아주는 방법으로 트레이드 오프를 선택