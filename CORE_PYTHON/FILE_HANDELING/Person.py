class Person:
    def __init__(self , fn , ln , age , g  ) :
        self.fname = fn 
        self.lname = ln
        self.age = age        
        self.gender = g 

    
    def display_info (self):
        print(f" Hi My Name is {self.fname} {self.lname} ,  \n I am {self.age} years \n I am {self.gender}") 