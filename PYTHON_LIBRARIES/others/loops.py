import numpy as np ; 
# a = np.arange(1, 25 ,1 ).reshape(6,4);
# for row in a :
#     for ele in row :
#          print(ele) ; 
# # using indexes 
# print (a) ; 

# for row_ind in range ( a.shape[0]) :
#      for col_ind in range ( a.shape[1]) : 
#           print(a[row_ind][col_ind] ,end=  " ")
#      print();     


# shortcut use np.nditer(a) to traverse throw any dimension array
# 
#  

# b = np.arange(1, 25 ,1 ).reshape(3 ,2 ,4);
# print(b); 
# for ele in np.nditer(b) : 
#      print(ele);


# a = np.arange(1 ,9).reshape(2 , 4 );
# print(a.shape);
# # print(a.T)
# # print(a);
# b = a.T
# print(b.shape);
# a = a.ravel()
# print(a.shape); 
 

# CONCEPT OF STACKING 
# a = np.arange(1 , 16 ).reshape(3,5); 
# b = np.arange(10 , 190, 10 ).reshape(3,6); 
# print(a)
# print(b)
# print(a.shape)
# print(b.shape)

# # c=np.hstack((a , b)) py loops.py

# # print(c)
# # print(c.shape)
# d = np.arange(10 , 310, 10 ).reshape(5,6); 
# print(d)
# print(d.shape)
# e=np.vstack((b,d))
# print(e)
# print(e.shape)

# arr = np.arange(10 ,160 ,10 ); 
# print(arr.shape);
# print(arr); 
# # you enter the array and the no of the parts 
# br = np.array_split(arr , 4)
# print(br); 
#   returned output is the list 
# print(br.shape)


# #  for cutting at the index you can pass the indexes 

# cr = np.split(arr , [3, 8])
# print(cr);

# splliting up of the 2 d array 
# ar = np.arange(10 ,190 ,10 ).reshape(3 ,6 ); 
# print(ar.shape);
# print(ar);
# dr = np.vsplit(ar ,3 ) 
# print(dr);

arr = np.arange(10 , 190 , 10 ).reshape(6,3 ); 
print (arr.shape)

print (arr[np.ix_([1,2,4],[1,2])])
# arr[[1,2,4]][:,[1,2]]
# print (arr[[1 ,  2 , 4] ][: , [  1 ,2]])
# print (arr[: ,[  1 ,2]])

#  boolean mask 

