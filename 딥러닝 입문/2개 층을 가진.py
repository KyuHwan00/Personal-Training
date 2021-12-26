import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split



class SingleLayer():
    def __init__(self, learning_rate = 0.1, l1 = 0, l2 = 0):
        self.w = None
        self.b = None
        self.lr = learning_rate
        self.w_history = []
        self.l1 = l1
        self.l2 = l2
        self.losses = []
        self.val_losses = []
    def activation(self, z):
        return 1/(1 + np.exp(-z))
   
    
    def forpass(self, x):
        z = np.dot(x,self.w) + self.b
        return z


    def backprop(self, x, err):
        m = len(x)
        grad_w = np.dot(x.T,err) / m
        grad_b = np.sum(err) / m
        return grad_w, grad_b
    
    def predict(self, x_t):
        z = self.forpass(x_t)
        return z > 0 
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y.reshape(-1,1))
    
    
    def update_val_loss(self, x_val, y_val):
        z = self.forpass(x_val)
        a = self.activation(z)
        a = np.clip(a , 1e-10, 1-1e-10)
        val_loss = np.sum(-(y_val*np.log(a) + (1-y_val)*np.log(1-a)))
        self.val_losses.append((val_loss + self.reg_loss())/ len(y_val) )    




    def reg_loss(self):
        return self.l1*np.sum(np.abs(self.w)) + self.l2*np.sum(self.w**2)/2

    
    def fit(self, x, y , epochs = 100, x_val = None, y_val = None):
        y = y.reshape(-1,1)
        y_val = y_val.reshape(-1,1)
        m = len(x)
        self.w = np.ones((x.shape[1],1))
        self.b = 0
        self.w_history.append(self.w.copy())
        for _ in range(epochs):
            z = self.forpass(x)
            a = self.activation(z)
            err = -(y-a)
            grad_w , grad_b = self.backprop(x, err)
            grad_w += (self.l1*np.sign(self.w) + self.l2*self.w) / m
            self.w -= self.lr*grad_w
            self.b -= self.lr*grad_b
            self.w_history.append(self.w.copy())
            a = np.clip(a, 1e-10, 1-1e-10)
            loss = np.sum(-(y*np.log(a) + (1-y)*np.log(1-a))) 
            self.losses.append((loss + self.reg_loss())/ m )
            self.update_val_loss(x_val, y_val)


class DualLayer(SingleLayer):
    def __init__(self, units = 10, learning_rate = 0.1, l1 = 0, l2 = 0):
        self.units = units
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.a1 = None
        self.losses =[]
        self.val_losses = []
        self.lr= learning_rate
        self.l1 = l1
        self.l2 = l2

    def forpass(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.activation(z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.z2
    def backprop(self, x, err):
        m = len(x)
        grad_w2 = np.dot(self.a1.T , err) / m
        grad_b2 = np.sum(err)/ m
        err_to_hidden = np.dot(err, self.w2.T)* self.a1 * (1-self.a1)
        grad_w1 = np.dot(x.T , err_to_hidden) / m
        grad_b1 = np.sum(err_to_hidden, axis = 0) / m
        return grad_w1, grad_b1 , grad_w2 , grad_b2
    def init_weights(self, n_features):
        self.w1 = np.ones((n_features, self.units))
        self.b1 = np.zeros(self.units)
        self.w2 = np.ones((self.units, 1))
        self.b2 = 0
    def fit(self, x,y, epochs =1000, x_val = None, y_val = None):
        y = y.reshape(-1, 1)
        y_val = y_val.reshape(-1,1)
        m = len(y)
        self.init_weights(x.shape[1])
        for _ in range(epochs):
            a = self.training(x,y,m)
            a = np.clip(a, 1e-10 , 1-1e-10)
            loss = np.sum(-(y*np.log(a) + (1-y)*np.log(1-a)))
            self.losses.append((loss + self.reg_loss()) / m)
            self.update_val_loss(x_val, y_val)

    def training(self, x , y, m):
        z = self.forpass(x)
        a = self.activation(z)
        err = -(y-a)
        grad_w1 ,grad_b1, grad_w2 , grad_b2 = self.backprop(x, err)
        grad_w1 += (self.l1*np.sign(self.w1) + self.l2*self.w1) / m
        grad_w2 += (self.l1*np.sign(self.w2) + self.l2*self.w2) / m
        self.w1 -= self.lr*grad_w1
        self.b1 -= self.lr*grad_b1
        self.w2 -= self.lr*grad_w2
        self.b2 -= self.lr*grad_b2
        return a
    def reg_loss(self):
        return self.l1*(np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))) + self.l2/2*(np.sum(self.w1**2) + np.sum(self.w2**2))

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
x_traing_all, x_test , y_traing_all, y_test = train_test_split(x, y, stratify= y, test_size= 0.2, random_state= 42)

x_traing, x_val, y_traing , y_val = train_test_split(x_traing_all, y_traing_all , stratify= y_traing_all, test_size= 0.2, random_state= 42)


from sklearn.preprocessing import StandardScaler

normal = StandardScaler()
normal.fit(x_traing)
x_train_scaled = normal.transform(x_traing)
x_val_scaled = normal.transform(x_val)

"""
dual_layer = DualLayer(l2= 0.01)
dual_layer.fit(x_train_scaled, y_traing, epochs = 20000,x_val = x_val_scaled , y_val = y_val)

dual_layer.score(x_val_scaled, y_val)

plt.ylim(0,0.3)
plt.plot(dual_layer.losses)
plt.plot(dual_layer.val_losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
"""

# 가중치 초기화하기
# 그래프의 곡선이 매끄럽지 않은 경우 => 손실함수가 감소하는 방향을 올바르게 찾는 데 시간이 많이 지체 => 초기 가중치 1의 값을 가진 행렬로 배치
# 해결방법 : 초기화시 정규분포를 따르는 무작위 수로 가중치 초기화 => np.random.seed(42) / np.random.normal(0[평균], 1[표준편차], (배치크기))
class RandomInitNetwork(DualLayer):
    def init_weights(self, n_features):
        np.random.seed(42)
        self.w1 = np.random.normal(0,1,(n_features, self.units))
        self.w2 = np.random.normal(0,1, (self.units, 1))
        self.b1 = np.zeros(self.units)
        self.b2 = 0

class MiniBatchNetwork(RandomInitNetwork):
    def __init__(self, batch_size = 32 , units=10, learning_rate=0.1, l1=0, l2=0):
        super().__init__(units=units, learning_rate=learning_rate, l1=l1, l2=l2)
        self.batch_size = batch_size
    def fit(self,x,y,epochs =100, x_val = None, y_val = None):
        y = y.reshape(-1,1)
        y_val = y_val.reshape(-1,1)
        np.random.seed(42)
        self.init_weights(x.shape[1])
        for _ in range(epochs):
            loss = 0
            for x_batch, y_batch in self.gen_batch(x,y):
                y_batch = y_batch.reshape(-1,1)
                m = len(x_batch)
                a = self.training(x_batch, y_batch, m)
                a = np.clip(a,1e-10, 1-1e-10)
                loss += np.sum(-(y_batch*np.log(a) + (1-y_batch)* np.log(1-a)))
            self.losses.append((loss + self.reg_loss()) / len(x))
            self.update_val_loss(x_val, y_val)

    def gen_batch(self, x, y):
        length = len(x)
        bins = length // self.batch_size
        if length % self.batch_size :
            bins +=1
        indexes = np.random.permutation(np.arange(length))
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start = self.batch_size*i
            end = self.batch_size*(i+1)
            yield x[start : end], y[start : end]

minibatch_net = MiniBatchNetwork(l2 =0.01, batch_size= 32)
minibatch_net.fit(x_train_scaled, y_traing, x_val = x_val_scaled, y_val = y_val)
minibatch_net.score(x_val_scaled, y_val)

plt.plot(minibatch_net.losses)
plt.plot(minibatch_net.val_losses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0,0.3)
plt.legend(['train', 'val'])
plt.show()