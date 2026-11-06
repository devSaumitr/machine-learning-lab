import numpy as np 

a = np.random.randint(1 ,10, 25)
print(a)
b = np.random.randint(1 ,10 , 25)
print(b)
c = np.union1d(a ,b)
print(c)

d =np.setdiff1d(a, b )
print(d)

e = np.intersect1d(a , b )
print(e)

f = np.setdiff1d(b , a )
print(f)

g = np.setxor1d(a , b )
print(g)

a = np.random.randint(10 , 100 , 24)
print(a)
print(np.clip(a ,a_min = 25 , a_max = 75))