from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

cancer = load_breast_cancer()
data = cancer.data
target = cancer.target

x_train, x_test , y_train , y_test = train_test_split(data, target, stratify= target, test_size= 0.2 , random_state= 42)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify= y_train, test_size=0.2, random_state= 42)

sgd = SGDClassifier(loss= 'log' , random_state= 42)
sgd.fit(x_train, y_train)
sgd.score(x_val, y_val)
