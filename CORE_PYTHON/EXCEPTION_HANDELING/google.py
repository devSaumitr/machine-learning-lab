class SecurityError(Exception):
    def __init__(self , message) :
        print(message)
    def logout(self) :
        print("logged out from every device ")

class Google :
    
    def __init__(self, name , email, password ,device ):
        self.name = name 
        self.email = email 
        self.password = password 
        self.device = device 
    def login(self , email , password, device) :
        if( device != self.device):
            raise SecurityError(" THE Device is not same") 
        if(email == self.email and password == self.password ) :
            print("you have succesfully logged in")
        else :
            print(" Login Error")
            
            
            
            
person = Google("saumitra" , "saumitr123@gmail.com" , "1234abcd" , "Android")
 
 
try :
    person.login("saumitr123@gmail.com" , "1234abcd" , "Android")
except SecurityError as e :
    print(e)                 
    e.logout() ; 
else :
    print(person.name)
finally: 
    print("database connection closed ")    