import numpy as np 

# a = np.arange(1 , 13 ) 
# print( a)
# print(a[2 : : -1])
# print( a[-5: : -1])
# print( a[-5 : ])
        # a = np.arange(1 , 13 ).reshape( 3 , 4 )
        # print(a[1][2])


        # a = np.arange(1 ,  28 ).reshape(3 ,3 ,3 )
        # print(a)
        # print(a[2][1][2])
        # print(a[0][1][0])

a = np.arange(1 , 64 ).reshape(2 , 4,2 , 2 )
print(a)
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