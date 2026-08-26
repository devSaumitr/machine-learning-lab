import numpy as np ; 
# # a = np.zeros(9).reshape((3,3));
# # print(a); 
# # print(np.shape(a));
# b = np.array([[1,2,3], [4,5,6], [7,8,9]])
# print(b); 
# print(b.shape);
# print(b[1][2])
# #  concept of slicing 
# a  = np.arange( 1 , 10  )
# print(a[1 : 5])
# b = np.arange( 1 , 10  )

# # print(a + b ); 
# # print(a - b ); 
# # print(a * b ); 
# # print(a / b ); 
# # print(a ^ b ); 


# print(np.random.random(5))
# print(np.random.randint(1 , 10 , 5))
# a = np.arange(1 , 10  , dtype= np.int32).reshape(3 ,3 ); 
# b = np.arange(1 , 10  , ).reshape(3 ,3 ); 
# c= np.arange(1 , 10  , dtype= np.float64).reshape(3 ,3 ); 
# print(a.itemsize);
# print(b.itemsize);
# print(c.itemsize);
# print(a.dtype);
# print(b.dtype);
# print(c.dtype);
# #  changing datatype
# c =c.astype(np.int16)
# print(c.dtype);
# import numpy as np

# # arr = np.array([1, 2, 3, 4])

# # new_arr = arr.astype(float)

# # print(new_arr) 
# # #  changing the  datatype  creates new array that need to  be stored it does not make the implicit changes 


# # print(arr > 4);
# arr = np.array([1, 2, 3])

# print(id(arr))

# arr = arr + 5
# print(id(arr))
# a = np.random.randint( 1 , 7777  , 15).reshape( 3,5); 
# print(a)
# print ();
# # a1 = np.sqrt(a  )
# # print(a1) 
# #  axis = 0 -> column 
# # axis = 1  -> row 

# print (np.min(a ,axis= 0 ));

a = np.random.randint(1 ,45 , 15).reshape(3 ,5 );
b = np.random.randint(4 ,75, 15).reshape(5 ,3 );
print(a);
print();
print(b);
print();
print ( np.dot(a , b))
#  stasttical fun on the  


