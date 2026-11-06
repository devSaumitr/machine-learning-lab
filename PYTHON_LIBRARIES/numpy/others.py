import numpy as np 

# l = [[1,2,3], [4,5,6]]
# a = np.array(l) 

# print(a)



# #  axis = 0 col and axis = 1 row 

# print( np.prod(a , axis = 0) , end  = "\n")

# print( np.prod(a , axis = 1) )


# print( np.mean(a , axis = 0))

# print( np.var(a , axis = 0))

# print( np.cos(a ))




# b = np.random.randint(low = 10 , high = 100 , size = ( 3 ,4 ))
# print (b)


# c= np.random.randint(low = 10 , high = 100 , size = ( 4 ,3 ))
# print (c)

# d = np.dot(b , c)
# print(d)

a = np.ceil(np.random.random((3 , 3  ,) ) * 100  ).astype(np.int32)
print(a)