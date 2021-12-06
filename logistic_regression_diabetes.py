import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#unpack data
np.set_printoptions(suppress=True)
df = pd.read_csv("diabetes_diagnosis_data.csv")
#split data into train, test
train = df.to_numpy(dtype="float16")[:-50]
test = df.to_numpy(dtype="float16")[-50:]

#define X (the inputs), and Y (the binary output)
'''
X:
0: # of pregnancies
1: glucose
2: blood pressure
3: skin thickness
4: insulin
5. BMI
6. diabetes pedigree function
7. age
Y:
0 --> no diagnosis
1 --> diabetes diagnosed
'''
X=np.ndarray([train.shape[0], train.shape[1]-1])
Y=np.ndarray([train.shape[0], 1])
for i in range(len(train)):
    X[i]=train[i][0:-1]
    Y[i]=train[i][-1]
    
#dimensions of matrices
m, n = X.shape
#learning rate
L = .0001
#weights, as a vector
W = np.zeros(n)
#bias
b=0

#train model
for epoch in range(1000):
    guess = 1/(1+np.exp(-(X.dot(W)+b)))
    diff = np.reshape(guess-Y.T, m)
    dW = np.dot(X.T, diff)/m
    db = np.sum(diff)/m
    
    W = W + dW*L
    b = b + db*L

    np.dot(-Y.T, np.log(guess))-np.dot(1-Y.T, np.log(1-guess))
    print(error)

#test model
accuracy = 0
for i in test:
    guess = round(sigmoid(np.dot(i[:-1], W.T)))
    print(i[-1]==guess)
    accuracy += (i[-1]==guess)
print(str(100*accuracy/len(test))+"%")
