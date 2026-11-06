#  ndim tells the no of hte dimension 
   
import numpy as np

# a = np.random.random(( 3 ,3 ,3) )
# print(a.ndim)
# print(a.shape)
# print(a.size)
# print(a.itemsize)


a =np.array( [[1,2,3],[4,5,6]]  )
print(a.ndim)
print(a.shape)
print(a.size)
print(a.itemsize)



b = a.astype(np.int32)
 
print(b.itemsize)