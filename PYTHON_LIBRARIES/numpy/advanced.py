import numpy as np 


# a = np.random.randint(10 , 50 , 24 ).reshape(6, 4 )


# print(a)
# # a = np.append(a , 200)

# a = np.append( a ,np.random.randint(100 , 300  , a.shape[0]).reshape(-1 , 1 ) , axis = 1  )
# print(a)


# a = np.random.randint(10 , 100 , 12).reshape(3 ,4 )
# b = np.random.randint(10 , 100 , 12).reshape(3 ,4 )

# c = np.concatenate((a , b ) , axis = 1 )

# c = np.concatenate((a , c) , axis = 1 )
# print(c)
# u = np.unique(c)
# print(u)
# # axis=0
# # ↓
# # ↓
# # ↓
# # rows direction

# # axis=1
# # → → →
# # columns direction

# # a = np.random.randint(10 , 100 , 10)
# # print(a.shape)
# # a = np.expand_dims(a , axis = 1)
# # print(a)
# # print(a.shape)

# # a = np.random.randint(10 , 100 , 10)
# # print(a.shape)
# # a = np.expand_dims(a , axis = 0)
# # print(a)
# # print(a.shape)

# # print(np.where(a > 50))
# # # when u need to perform any operation on the values

# # # syntax np.where(condtion , true , false )

# # # a = np.where(a > 50 , 0 , a)
# # print(a);

# # print(np.argmax(a ,axis = 0 ))





# a = np.random.randint(10 , 50 , 24 ).reshape(6, 4)
# print(a)
# print(np.argmax(a , axis = 0))

# print(np.argmax(a , axis = 1))


# print(np.argmin(a , axis = 0))

# print(np.argmin(a , axis = 1))
# a =  np.random.randint( 10 , 50 , 24).reshape(6, 4)
# b = np.cumsum(a ,axis = 0 ) 
# print(a , "\n" , b)

# c = np.cumprod(a , axis = 1)
# print(a , "\n"  ,c)

# a = np.random.randint( 10 , 100 , 34)
# print(a)
# print(np.percentile(a , 100 ) )



# a = np.random.randint( 10 , 100 , 34)
# print(a) 
# print(np.histogram(a , bins = [1,10 ,20,30,40,50,60,70,80,90,100])
# )



# salary = np.random.randint(10 , 100 , 15)
# exp = np.random.randint(1 ,5 , 15)
# print(np.corrcoef(salary  , exp ))



# a = np.random.randint(9 , 1000 , 23)
# items = [10 , 20 ,30 ,40 ,50, 60 , 70 , 80 ,90, 100]
# print(a[np.isin(a , items)])



# a = np.random.randint(9 , 1000 , 24).reshape(6, 4)
# print(a)
# print(np.flip(a))



# a = np.random.randint(9 , 1000 , 23)
# print(a)
# np.put(a , [2,3,5,6] , [100,200,300,400])
# print(a)


a = np.random.randint(9 , 1000 , 23)


print(a)
np.delete(a , [4,6,7])

print(a)