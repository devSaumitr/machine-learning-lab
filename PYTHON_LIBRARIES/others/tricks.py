import numpy as np;
# arr = np.random.randint(1, 100, 10);
# print(arr);
# arr = np.sort(arr)[::-1];
# print(arr);
# arr = np.append(arr, 100);
# print(arr);
# br = np.arange(10, 210, 10).reshape(4,5);
# print(br);
# br = np.append(br ,[[np.random.randint(1, 100, 5)]] ,axis=0);
# print(br);



# br = np.arange(10, 210, 10).reshape(4,5)
# print(br)

# br = np.append(br, [np.random.randint(1, 100, size=(1, 5))], axis=0)
# print(br)



# br = np.arange(10, 210, 10).reshape(4,5)
# print(br)
# new_col = np.random.randint(1, 100,size = (4 ,1 )) 
# br = np.append(br, new_col , axis =1) 

# print(br)



# # ar = np.arange ( 10 , 160 , 10 ).reshape(5 ,3 )
# # print(ar.shape);
# # print (ar );
# # br = np.arange(200 , 350 , 10 ).reshape(5 ,3 )
# # print(br.shape);
# # print (br );
# # # axis = 0 means row wise concatenation , vertical concatenation
# # c =np.concatenate ((ar , br ), axis = 0 ) 
# # # axis = 1 means column wise concatenation , horizontal concatenation
# # d =np.concatenate ((ar , br ), axis = 1) 
# # print(c.shape);
# # print (c);
# # print (d.shape);
# # print ( d) py ; 


# # ar = np.arange(10 , 160 , 10 ); 
# # print (ar .shape); 
# # br = np.expand_dims(ar , axis = 0 ); 
# # cr = np.expand_dims(ar , axis = 1 ); 
# # print ( br.shape);
# # print (br)
# # print (cr.shape);
# # print (cr);
# # dr = np.expand_dims(br , axis = 2 ); 
# # print (dr.shape) ;
# # print ( dr ); 


# # ar = np.arange(10 , 160 , 10 ) ; 
# # br = np.where(ar > 40 );
# # print (br);
# # cr = np.where(((ar > 79) & (ar < 120)) , "yes" , "no" );
# # print (cr);







# # ar = np.arange(10 , 160 , 10 );
# # print (np.argmax(ar));
# br = np.random.randint( 10 , 100 ,15).reshape(5 ,3 );
# print(br);
# # print(np.argmax(br ,axis = 0 ));
# # print(np.max(br ,axis = 0 ));
# # print(np.max(br ,axis = 1 ));
# # print(np.argmax(br ,axis = 1 ));
# print(np.cumsum(br  , axis = 1));
# # ar = np.arange(10 , 160 , 10 );


# ar = np.random.randint( 10 , 100 ,15);
# print(ar);
# br =np.array([68, 95, 47, 73, 50, 27, 34, 65, 57, 29, 31, 81, 44, 18, 15]);
# print(np.percentile(br , 95)); 

br =np.array([68, 95, 47, 73, 50, 27, 34, 65, 57, 29, 31, 81, 44, 18, 15]);

count , edges = np.histogram(br , bins = 7 ); 

print(count);
print ("=================================")
print(edges);
print ("=================================")
print(edges[1] -edges[0])

print(np.diff(edges))