import numpy as np 


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


class NeuaraNetwork:
    def __init__(self,x ,y,num_of_hidden_layers) -> None:
        self.x = x
        self.y = y.T
        self.neurons=num_of_hidden_layers
        np.random.seed(1)
        self.w1 = np.random.rand(self.x.shape[1],self.neurons)
        np.random.seed(1)
        self.w2 = np.random.rand(self.neurons,1)

    def forward(self):
        self.layer1 = sigmoid(np.dot(self.x,self.w1))
        self.output = sigmoid(np.dot(self.layer1,self.w2))
    

    def back(self):
        d_w2 = np.dot(self.layer1.T,(2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_w1 = np.dot(self.x.T,
                      (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output),
                              self.w2.T)* sigmoid_derivative(self.layer1)))
        self.w1 += (d_w1)
        self.w2 += (d_w2)
        
        
    def predict(self):
        self.layer1 = sigmoid(np.dot(self.x,self.w1))
        self.output = sigmoid(np.dot(self.layer1,self.w2))
        return self.output
        
        
        
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0,1,1,0]])
nn = NeuaraNetwork(x, y,2)


for i in range(10000):
    nn.forward()
    nn.back()
    h = nn.predict()

print('y')
print(y.T)
print('h')
print(h)

