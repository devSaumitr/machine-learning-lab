import numpy as np 
# a = np.arange(1 , 33 ).reshape(4,2,4)
# print(a)
# # print(a[1][0][1][1])

# print(a[1::2 , :: ,1:3 ])
# print(a[::2 , 0 , ::2])
# # \ iterations in numpy 


# for i in np.nditer(a):
#     print(i) 
b= np.arange(1 ,13).reshape(4 ,3)
    
# print(b , end = "m ") ;

d = b.ravel()
print(d) 
c = np.transpose(b)
print(c)