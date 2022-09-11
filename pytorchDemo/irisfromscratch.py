import numpy as np
# Creating data set
InputLayerN = 4
HiddenLayerN = 6
OutputLayerN = 4


# A
a =[1, 0, 0, 0]
# B
b =[0, 1, 0, 0]
# C
c =[0, 1, 0, 1]

d =[0, 1, 0, 1]

# Creating labels
y =[[1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1]]

# converting data and labels into numpy array
 
"""
Convert the matrix of 0 and 1 into one hot vector
so that we can directly feed it to the neural network,
these vectors are then stored in a list x.
"""
 
x =[np.array(a).reshape(1, InputLayerN),
    np.array(b).reshape(1, InputLayerN),
    np.array(c).reshape(1, InputLayerN),
    np.array(d).reshape(1, InputLayerN)]
 
 
# Labels are also converted into NumPy array
y = np.array(y)
 
 
#print(x, "\nand\n", y)

# activation function
def sigmoid(x):
    return(1/(1 + np.exp(-x)))
   
def f_forward(x, w1, w2, w3):
    # hidden

    z1 = x.dot(w1)# input from layer 1
    a1 = sigmoid(z1)# out put of layer 2


    z2 = a1.dot(w2)# input from layer 1
    a2 = sigmoid(z2)# out put of layer 2

     
    # Output layer
    z3 = a2.dot(w3)# input of out layer
    a3 = sigmoid(z3)# output of out layer
    return(a3)
  
# initializing the weights randomly
def generate_wt(x, y):
    l =[]
    for i in range(x * y):
        l.append(1)
    return(np.array(l).reshape(x, y))
     
# for loss we will be using mean square error(MSE)
def loss(out, Y):
    s =(np.square(out-Y))
    s = np.sum(s)/len(y)
    return(s)
   
# Back propagation of error
def back_prop(x, y, w1, w2, w3, alpha, i):
    # hidden layer
    z1 = x.dot(w1)# input from layer 1
    a1 = sigmoid(z1)# output of layer 2
    
    # Output layer
    z2 = a1.dot(w2)# input of out layer
    a2 = sigmoid(z2)# output of out layer
    
    # Output layer
    z3 = a2.dot(w3)# input of out layer
    a3 = sigmoid(z3)# output of out layer
    
    # error in output layer
    # d3 =(a3-y)
    l2 = (a3-y)
    d3 = np.multiply(l2, (np.multiply(a3, 1-a3))) 
    
    d2 = np.multiply( (w3.dot((d3.transpose()))).transpose(), (np.multiply(a2, 1-a2)) )
    d1 = np.multiply((w2.dot((d2.transpose()))).transpose(),(np.multiply(a1, 1-a1)))
     
    # Gradient for w1 and w3
    w1_adj = x.transpose().dot(d1)
    w2_adj = a1.transpose().dot(d2)
    w3_adj = a2.transpose().dot(d3)
    
    w1  = w1-(alpha*(w1_adj))
    w2 = w2-(alpha*(w2_adj))
    w3  = w3-(alpha*(w3_adj))

    
    return(w1,w2, w3)
 
def train(x, Y, w1,w2, w3, alpha = 0.01, epoch = 10):
    acc =[]
    losss =[]
    for j in range(epoch):
        l =[]
        for i in range(len(x)):
            out = f_forward(x[i], w1,w2, w3)
            l.append((loss(out, Y[i])))
            w1,w2, w3 = back_prop(x[i], y[i], w1,w2, w3, alpha, i)
        print("epochs:", j + 1, "======== acc:", (1-(sum(l)/len(x)))*100)  
        acc.append((1-(sum(l)/len(x)))*100)
        losss.append(sum(l)/len(x))
    return(acc, losss, w1, w2, w3)
  
    
w1 = generate_wt(InputLayerN, HiddenLayerN)
w2 = generate_wt(HiddenLayerN, OutputLayerN)
w3 = generate_wt(OutputLayerN, 3)

acc, losss, w1, w2, w3 = train(x, y, w1, w2, w3, 0.1, 100)
