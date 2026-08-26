#  pyhthon is slow
# buts its really easy to learn 
# so numpy as was  created in c and used in pyhton gave speed o0f the  c and ease  of pyhton




import numpy as np  
a =np.array([1,2,3])
print(a)
print(a.sum()) 

b = np.array([[1,2,3],[3,4,5],[5,6,7]] , dtype =complex)
print(b)
 
 
c = np.arange(1 , 17 , 1).reshape(2,4, 2 )
print(c)


d = np.ones((3 , 4) )
print(d)
e = np.zeros((2 , 4 ))
print(e)

f = np.random.random((3 , 4 ))
print(f)
g = np.linspace( -10 , 10 , 12, dtype = int ).reshape(3 ,4 )
print(g)
h= np.identity((3 ))
print(h)