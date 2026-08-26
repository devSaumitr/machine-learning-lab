import numpy as np ; 
ar = np.arange(10 , 160 , 10 ).reshape(5, 3 )
print (ar.shape); 
print ("=================================")
print (ar);
print ("=================================")

# #  case 1 mask 
# print (ar > 40 ); 
# #  case 2 mask 
# print ("=================================")
# print ((ar > 40) & (ar < 70 )); 
# print ("=================================")

# print ((ar > 40) | (ar < 70 )); 
# print ("=================================")
 
# br = ar [(ar > 50) & (ar < 110)]
# # br =  ar[ar > 40] ; 
# print(br);
# print ("=================================")
# cr = ar[br>= 90]
# # cr =np.array(cr);
# print(cr);
# cr  = cr [ :,[0] ]
# print ("=================================")
# print(

# br = ar[: , 0 ]   
# cr = ar[br > 40]
# br = ar [ar[: , 0 ]> 40 ]

# print ( cr );
# print ("=================================")
# print ( br );
# print ("=================================")
# dr = ar[: , 0]
# print(dr) ;
# print ("=================================")
# dr = ar [: , 0][ar [:  , 0 ] > 40 ]
# print(dr) ;
print ("=================================")
msk = ar[ : ,0 ] > 70 ;
print(msk)

print ("=================================")
dr = (ar[:][msk])
print(dr)

print ("=================================")
er = (ar[:, 0 ][ar[ : ,0 ] > 70 ])
print(er)

# 1. If mask.shape == array.shape
#    → select individual elements.

# 2. If a 1D mask is applied to a 2D array and its length
#    equals the number of rows
#    → select rows.

# 3. If you first select a row/column/slice and then apply
#    the mask
#    → the mask works only on that selected portion.