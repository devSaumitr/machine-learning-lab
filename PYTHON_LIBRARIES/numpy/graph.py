import numpy as np 
x = np.linspace( -10 , 10 , 15)
y = x**2 

def sigmoid(a) :
    return 1 / ( 1 + np.exp(-a));

y = (sigmoid(x))



import matplotlib.pyplot as plt 
plt.plot(x , y ) 
plt.show()

