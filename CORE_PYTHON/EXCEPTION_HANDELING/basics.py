     
     
try:
        with open ("sample.txt" , "r") as f :
            d = f.read()        
except FileNotFoundError :
    print("File Not Found "); 
except  NameError :
    print("var is not declared")    
except ZeroDivisionError :
    print(" division by 0 not possilble")                     
except Exception as e   : 
    print(f"some error occured {f}")
else :
    
            print(d)  ;
finally :
    print(("this will execute irrespective of any thing "))                
    
    
    
    