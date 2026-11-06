import json


# # l = [1,2,3,4]
# dic = {
#     'name' : "saumitra " ,
#     "Age" : 23 
# }

# with open ("demo.json" , "w") as f :
#     json.dump( dic  , f , indent = 4 )
    
    
   
    
# with  open("demo.json" , "r") as f : 
#      d = (json.load(f))    
#      print(d)
#      print(type(d))
     
     
     
     
     
    #   what to do in case of the tuples 
    
    # you cannot store the tuples in the json 
    # on retrival it gives the list   file 
    
    
tup = (1,2,3,4,5,6,7,8,9)

with open ("demo1.json" , "w") as f : 
        json.dump(tup , f , indent = 4 )
        
        
with open ("demo1.json" , 'r') as f :
    d = json.load(f)
    print(d)
    print(type(d))        