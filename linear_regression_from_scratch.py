from os import name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,confusion_matrix


class Regression:
    theta = np.matrix(np.zeros(1))


    def data_normalize(data):
        return ((data-data.mean())/data.std()).round(4)

    def rescale(data):
        return (data/data.max()).round(4)

    def data_preparing(self,X,y):
        df = pd.DataFrame(X)
        df.insert(0,'ones',1)
        X = np.matrix(df.iloc[:,:])
        y = np.matrix(y)
        # X = X/X.max()
        # y = y/y.max()
        return X,y

    
    def compute_cost(self,X,y,theta):
        z = np.power(((X*theta.T) - y),2)
        return np.sum(z)/(2*len(X))

    def fit(self,X,y):
        iterations=10000000
        alpha = 0.001
        X,y = self.data_preparing(X,y)
        theta = np.matrix(np.zeros(X.shape[1]))
        temp = np.matrix(np.zeros(theta.shape))
        cost = np.zeros(iterations)
        for i in range(iterations):
            error = (X*theta.T)-y
            for j in range(X.shape[1]):
                term = np.multiply(error,X[:,j])
                temp[0,j] = theta[0,j]-( (alpha/len(X)) * (np.sum(term)) )
                theta = temp
                self.theta = temp
            cost[i] = self.compute_cost(X, y, theta)
            if math.isclose(cost[i],cost[i-1]):
                iterations = i
                break
        #gradent descent graph
        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(cost[:iterations],'r')
        plt.show()
        return True

    def predict(self,X_test):
        df = pd.DataFrame(X_test)
        df.insert(0,'ones',1)
        X = np.matrix(df.iloc[:,:])
        return X*self.theta.T



#reading dataset
data=pd.read_csv('data.csv')
# get x and y
cols = data.shape[1]
X = data.iloc[:,0:cols-1].values
y = data.iloc[:,cols-1:cols].values
#spliting training and testing x,y
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
#train the model
regression = Regression()
regression.fit(x_train,y_train)
#testing
y_pred = regression.predict(x_test)
#dataset visualization
plt.scatter(x_train,y_train,color = 'red')
plt.scatter(x_test,y_test,color = 'green')
plt.plot(x_train,regression.predict(x_train),color = 'blue')
plt.title('plot')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
def data_normalize(data):
        return ((data-data.mean())/data.std()).round(4)