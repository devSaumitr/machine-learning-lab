dict = {
    'name' : "saumitra " ,
    "Age" : 23 
}

with open("test.txt" , 'r') as f :
    f.write(str(dict)) 
    
    
    
    #  you can store ony strings inside the txt file for binary files you can use the binary mode but for the  complex data structues u cant store the information 
    
    # this problem is solved by the serialization and deserialization
    
    
    # 