import numpy as np 

a= np.arange(1 , 13 ).reshape(4,3)
b= np.arange(13 , 25).reshape(4 ,3 )

# c = np.hstack((a , b ,a) )
# # //for horizontal stacking  no of the rows must be the same 
# c = np.hstack((c , a))
# c = np.vstack((a , b ,a) )
# c = np.vstack((c , b))
# print(c)

# //for vertical stacking  no of the columns must be the same








# #  NOW SPLITTING 
#  cutting vertically sepreates horizontal array called as hsplit 
#  cutting horizontally seprates the vertical array called as the vsplit 
a = np.hsplit(a , 3  )
print(a)

b =np.vsplit(b , 2 )
print(b)